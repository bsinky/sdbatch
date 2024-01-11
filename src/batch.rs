use self::auto1111_api::APIClient;
use choose_rand::rand::{ChooseRand, Probable};
use chrono::Local;
use image::io::Reader as ImageReader;
use rand::Rng;
use rand::{seq::SliceRandom, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::{
    fmt::{self},
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
};

mod auto1111_api;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct PromptData {
    positive: String,
    negative: String,
    model: String,
    /// Sampler to use, ex., "DPM 2M++ Karras"
    sampler: String,
    steps: u32,
    /// Image width at generation time, before Hi-res
    width: u32,
    /// Image height at generation time, before Hi-res
    height: u32,
    cfg: f32,
    // Clip Skip setting, defaults to 1
    clip_skip: Option<u8>,
    seed: Option<i64>,
    hires: Option<HiResSettings>,
    /// Post-processing to perform on generated image
    post_process: Option<PostProcesses>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct HiResSettings {
    upscaler: String,
    upscale_by: f32,
    denoising_strength: f32,
    steps: u8,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum PostProcesses {
    Resize { scale_by: f32 },
}

#[derive(Serialize, Deserialize, Default)]
pub struct BatchTemplate {
    /// Short name, used in the filename for logs after template runs
    pub name: String,

    /// Longer, detailed description
    pub description: Option<String>,

    /// Automatic1111 URL
    ///
    /// Defaults to http://127.0.0.1:7860
    pub api_url: Option<String>,

    /// Prompt setup to be used for all images
    pub base_prompt: PromptData,

    /// Number of images to generate,
    /// must be equal to or smaller than the number of prompts
    ///
    /// Defaults to the number of prompts in the pool
    pub count: Option<usize>,

    /// Whether Automatic1111 should save images on the server, defaults to false
    pub save_images: Option<bool>,

    /// Whether Automatic1111 should run face restoration, defaults to false
    pub restore_faces: Option<bool>,

    /// The pool of prompts to pick from
    pub prompts: Vec<Prompts>,

    /// Additional modifiers to add to each prompt
    pub modifiers: Option<Vec<PromptModifer>>,
}

fn get_api_client(
    api_url: Option<&str>,
    save_images: &Option<bool>,
    restore_faces: &Option<bool>,
) -> anyhow::Result<APIClient> {
    let url_to_use = match api_url {
        Some(url) => url,
        None => "http://127.0.0.1:7860",
    };
    println!("Using API at: {}", url_to_use);
    let api = auto1111_api::APIClient::new(&url_to_use, save_images, restore_faces)?;
    Ok(api)
}

#[derive(Debug, Clone)]
pub struct BatchError {
    message: String,
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unable to run template, {}", self.message)
    }
}

impl std::error::Error for BatchError {}

#[derive(Deserialize)]
struct Txt2ImgInfo {
    all_seeds: Vec<i64>,
}

impl BatchTemplate {
    fn run(
        &self,
        dry_run: bool,
        output_dir: &Path,
        sequential: bool,
        api_url: Option<&str>,
    ) -> anyhow::Result<BatchLog> {
        let count = self.count.unwrap_or(self.prompts.len());
        if self.prompts.len() < count {
            return Err(BatchError {
                message:
                    "count is too large, it must be less than or equal to the number of prompts"
                        .to_owned(),
            }
            .into());
        }

        let prompt_pool: Vec<&Prompts> = if sequential {
            self.prompts.iter().take(count).collect()
        } else {
            self.prompts
                .choose_multiple(&mut rand::thread_rng(), count)
                .collect()
        };

        let mut batch_log = BatchLog::new(&self.name, output_dir);
        std::fs::create_dir_all(output_dir)?;

        for prompt in prompt_pool.iter() {
            let prompt_data = self.generate_log_for_prompt(prompt);
            batch_log.images.push(prompt_data);
        }
        let batch_log_name = batch_log.write()?;
        let batch_log_name = batch_log_name
            .file_name()
            .expect("log file to have a valid filename");

        if dry_run {
            println!("Created log file {}", batch_log_name.to_string_lossy());
        } else {
            let api = get_api_client(api_url, &self.save_images, &self.restore_faces)?;
            println!(
                "Created log file {}, beginning image generation...",
                batch_log_name.to_string_lossy()
            );
            for (prompt_index, prompt) in batch_log.images.iter_mut().enumerate() {
                println!("Generating image {} of {}...", prompt_index + 1, count);
                Self::generate_image(output_dir, &api, prompt, prompt_index)?;
            }
        }

        Ok(batch_log)
    }

    /// Build a full positive prompt using Template's positive prompt settings and the given positive fragment
    fn build_positive(&self, positive: &str) -> String {
        return Self::combine_prompts(&self.base_prompt.positive, positive);
    }

    fn combine_prompts(a: &str, b: &str) -> String {
        let mut combined = a.trim_end().to_string();

        if combined.chars().last() != Some(',') {
            combined.push(',');
        }

        combined.push(' ');
        combined.push_str(b);
        return combined;
    }

    /// Copy template's base prompt and use the given positive prompt fragment to construct the positive prompt
    fn copy_with_positive(&self, positive: &str) -> PromptData {
        let mut data = self.base_prompt.clone();
        data.positive = self.build_positive(positive);
        return data;
    }

    fn generate_log_for_prompt(&self, prompt: &Prompts) -> PromptData {
        let mut rng = rand::thread_rng();
        let mut prompt_data = match prompt {
            Prompts::Single(positive) => self.copy_with_positive(positive),
            Prompts::Multiple(positive_vec) => {
                let positive = positive_vec.choose(&mut rng);
                self.copy_with_positive(
                    positive.expect("Prompts::Multiple to always pick Some positive prompt"),
                )
            }
            Prompts::MultipleWeighted(positive_vec) => {
                let v: Vec<_> = choose_rand::helper::refcellify(positive_vec.to_owned()).collect();

                let selected_prompt = v.choose_rand(&mut rng).expect("chances to sum to 1.0");
                self.copy_with_positive(&selected_prompt.prompt)
            }
        };

        if let Some(modifiers) = &self.modifiers {
            let applicable_modifiers: Vec<_> = modifiers
                .iter()
                .filter(|m| {
                    m.if_activator.is_none()
                        || m.if_activator
                            .as_ref()
                            .is_some_and(|activator| *&prompt_data.positive.contains(*&activator))
                })
                .filter(|m| {
                    m.if_not_activator.is_none()
                        || m.if_not_activator
                            .as_ref()
                            .is_some_and(|activator| filter_if_not(&prompt_data, &activator))
                })
                .collect();
            if let Some(modifier) = applicable_modifiers.choose(&mut rng) {
                let roll: f32 = rng.gen();
                if roll <= modifier.chance.unwrap_or(1.0) {
                    // ring-a-ding-ding!
                    prompt_data.positive =
                        Self::combine_prompts(&prompt_data.positive, &modifier.prompt);
                }
            }
        }

        // Assign a seed value
        prompt_data.seed = Some(rng.next_u32() as i64);

        return prompt_data;
    }

    /// Use the Automatic1111 API and generate an image for the given prompt, and sets the seed
    fn generate_image(
        output_dir: &Path,
        api: &auto1111_api::APIClient,
        prompt: &mut PromptData,
        prompt_index: usize,
    ) -> anyhow::Result<()> {
        let (image_list, info) = api.txt2img(prompt)?;

        let info: Txt2ImgInfo = serde_json::from_str(&info)?;

        for (i, image_bytes) in image_list.iter().enumerate() {
            let mut image_filename = PathBuf::from(output_dir);
            if i == 0 {
                image_filename.push(format!("{:02}.png", prompt_index));
                if let Some(seed) = info.all_seeds.first() {
                    prompt.seed = Some(*seed);
                }
            } else {
                image_filename.push(format!("{:02}-{}.png", prompt_index, i));
                // TODO: Better support for multiple images/batches.
                // TODO: We will have seeds for these, but nowhere to put them in the log.
            }

            match &prompt.post_process {
                None => {
                    let mut dest_file = fs::File::create(&image_filename)?;
                    dest_file.write_all(image_bytes)?;
                }
                Some(p) => {
                    print!("Post-processing...");
                    match p {
                        PostProcesses::Resize { scale_by } => {
                            let (orig_img_w, orig_img_h) = match &prompt.hires {
                                Some(hires) => (
                                    (prompt.width as f32 * hires.upscale_by) as u32,
                                    (prompt.height as f32 * hires.upscale_by) as u32,
                                ),
                                None => (prompt.width, prompt.height),
                            };
                            let orig_img = ImageReader::new(Cursor::new(image_bytes))
                                .with_guessed_format()?
                                .decode()?;
                            let new_w = (orig_img_w as f32 * scale_by) as u32;
                            let new_h = (orig_img_h as f32 * scale_by) as u32;
                            println!("resizing to {}x{}", new_w, new_h);
                            let resized_img = image::imageops::resize(
                                &orig_img,
                                new_w,
                                new_h,
                                image::imageops::FilterType::Lanczos3,
                            );
                            resized_img.save(&image_filename)?;
                        }
                    };
                }
            }
        }

        Ok(())
    }

    fn safe_template_filename(&self) -> PathBuf {
        let sanitized_filename = get_safe_filename(&self.name);
        return format!("{}.json", sanitized_filename).into();
    }

    /// Serialize to JSON and write to disk
    pub fn write(&self, output_dir: &Path) -> anyhow::Result<PathBuf> {
        let dest_filename = self.safe_template_filename();
        let (_, dest_file) = create_file_and_dir(output_dir, &dest_filename)?;
        serde_json::to_writer_pretty(&dest_file, self)?;

        Ok(dest_filename)
    }
}

fn filter_if_not(prompt: &PromptData, activator: &&OneToManyPrompts) -> bool {
    match activator {
        OneToManyPrompts::One(not_keyword) => {
            return !prompt.positive.contains(not_keyword)
        },
        OneToManyPrompts::Many(not_keywords) => {
            return !not_keywords.iter()
                .any(|keyword| prompt.positive.contains(keyword))
        },
    }
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum Prompts {
    /// Basic static prompt option
    ///
    /// ex., "1girl, solo, brown hair, brown eyes"
    Single(String),
    /// Multiple prompt options available, for different outfits, etc.
    /// One will be selected at random if this prompt is picked from the pool
    ///
    /// ex., \["1girl, solo, dress", "1girl, solo, shirt, jeans"\]
    Multiple(Vec<String>),
    /// Like Multiple, but with some options more likely to be picked than others
    /// The sum of the specified chances must add up to 1.0
    MultipleWeighted(Vec<WeightedPrompt>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WeightedPrompt {
    /// Prompt string to use
    prompt: String,
    /// The % chance for this prompt to be picked, defaults to 1.0
    ///
    /// ex., "0.8" would be an 80% chance
    chance: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OneToManyPrompts
{
    One(String),
    Many(Vec<String>)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PromptModifer {
    /// Prompt string to use
    prompt: String,
    /// The % chance for this prompt to be picked, defaults to 1.0
    ///
    /// ex., "0.8" would be an 80% chance
    chance: Option<f32>,
    /// If set, the modifier will only be considered if the selected prompt already contains the given string
    #[serde(rename = "if")]
    if_activator: Option<String>,
    /// If set, the modifier will only be considered if the selected prompt does not contain given string(s)
    #[serde(rename = "if-not")]
    if_not_activator: Option<OneToManyPrompts>,
}

impl Probable for WeightedPrompt {
    fn probability(&self) -> f32 {
        self.chance.unwrap_or(1.0)
    }
}

#[derive(Serialize, Deserialize)]
pub struct BatchLog {
    /// Template name used for generation
    template: String,

    /// Generated images
    images: Vec<PromptData>,

    #[serde(skip)]
    file_path: PathBuf,
}

fn create_file_and_dir(
    output_dir: &Path,
    dest_filename: &Path,
) -> anyhow::Result<(PathBuf, fs::File)> {
    let dest: PathBuf = PathBuf::from(output_dir).join(&dest_filename);
    std::fs::create_dir_all(output_dir)?;
    let dest_file = fs::File::create(&dest)?;
    Ok((dest, dest_file))
}

fn get_safe_filename(name: &str) -> String {
    // TODO: could do more to strip unsafe characters, but this is good enough for now
    let sanitized_filename = name.replace(" ", "-");
    return sanitized_filename;
}

impl BatchLog {
    fn new(name: &str, output_dir: &Path) -> BatchLog {
        let mut file_path = PathBuf::from(output_dir);
        file_path.push(Self::safe_logfile_name(name));
        return BatchLog {
            template: name.to_owned(),
            images: vec![],
            file_path
        };
    }

    fn from_file(file_path: &str) -> anyhow::Result<BatchLog> {
        let log_file = fs::File::open(file_path)?;
        let mut log: BatchLog = serde_json::from_reader(log_file)?;
        log.file_path = PathBuf::from_str(file_path)?;
        Ok(log)
    }

    fn safe_logfile_name(name: &str) -> PathBuf {
        let timestamp = Local::now();
        let timestamp = format!("{}", timestamp.format("%Y-%m-%d-%H%M"));

        let sanitized_filename = get_safe_filename(name);
        return format!("{}-{}.json", sanitized_filename, timestamp).into();
    }

    /// Serialize to JSON and write to disk
    pub fn write(&self) -> anyhow::Result<PathBuf> {
        let output_dir = &self.file_path.parent();
        if output_dir.is_none() {
            return Err(BatchError {
                message: "Invalid output directory for log".to_string(),
            }
            .into());
        }
        std::fs::create_dir_all(output_dir.unwrap())?;
        let dest_file = fs::File::create(&self.file_path)?;
        serde_json::to_writer_pretty(&dest_file, self)?;

        Ok(self.file_path.clone())
    }

    /// Serialize to JSON and write to disk, overwriting original file
    pub fn write_update(&self, log_file: &fs::File) -> anyhow::Result<()> {
        serde_json::to_writer_pretty(log_file, self)?;
        Ok(())
    }
}

pub struct TemplateRunResults {
    pub images_created: usize,
    pub log_file: PathBuf,
}

pub fn do_run(
    dry_run: bool,
    sequential: bool,
    template_filename: &str,
    output_dir: &str,
    api_url: Option<&str>,
) -> anyhow::Result<TemplateRunResults> {
    let json_string = fs::read_to_string(template_filename)?;
    let template: BatchTemplate = serde_json::from_str(&json_string)?;

    let output_dir = PathBuf::from(output_dir);

    let batch_log = template.run(dry_run, &output_dir, sequential, api_url)?;
    // let log_file = batch_log.write(&output_dir, &template.name)?;

    Ok(TemplateRunResults {
        images_created: batch_log.images.len(),
        log_file: batch_log.file_path,
    })
}

fn print_reroll_start(index: usize) {
    println!("Regenerating image {} with a new seed...", index);
}

pub fn reroll(file_path: &str, index: usize, api_url: Option<&str>) -> anyhow::Result<()> {
    let mut log = BatchLog::from_file(file_path)?;

    let path = PathBuf::from(file_path);
    let output_dir = path.parent().expect("couldn't get folder from file_path");

    match log.images.iter().nth(index) {
        None => Err(BatchError {
            message: format!(
                "Reroll error: Log file contains less than {} images",
                index + 1
            ),
        }
        .into()),
        Some(prompt) => {
            let api = get_api_client(api_url, &None, &None)?;

            let mut updated_prompt = prompt.to_owned();
            updated_prompt.seed = None;
            print_reroll_start(index);
            BatchTemplate::generate_image(output_dir, &api, &mut updated_prompt, index)?;
            log.images[index] = updated_prompt;
            let dest_file = fs::File::create(path)?;
            log.write_update(&dest_file)?;

            Ok(())
        }
    }
}

pub fn resume(file_path: &str, api_url: Option<&str>) -> anyhow::Result<u32> {
    let mut missing_images_created: u32 = 0;
    let mut log = BatchLog::from_file(file_path)?;

    let path = PathBuf::from(file_path);
    let output_dir = path.parent().expect("couldn't get folder from file_path");

    let api = get_api_client(api_url, &None, &None)?;

    let mut output_dir_entries = fs::read_dir(output_dir)?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, std::io::Error>>()?;

    output_dir_entries.sort();
    let mut existing_image_indices = vec![];
    for file_path in output_dir_entries {
        match file_path.extension() {
            None => {
                continue;
            }
            Some(ext) => {
                if ext != "png" {
                    continue;
                }
            }
        }

        if let Some(file_name) = file_path.file_stem() {
            if let Some(image_index) =
                usize::from_str_radix(file_name.to_str().unwrap_or(""), 10).ok()
            {
                existing_image_indices.push(image_index);
            }
        }
    }

    let mut first = true;
    for (index, prompt) in log.images.clone().iter().enumerate() {
        if existing_image_indices.contains(&index) {
            continue;
        } else if first {
            println!("Resuming starting with first missing image: {}", index);
            first = false;
        }
        println!("Generating image {}...", index);

        let mut updated_prompt = prompt.to_owned();
        updated_prompt.seed = None;
        BatchTemplate::generate_image(output_dir, &api, &mut updated_prompt, index)?;
        log.images[index] = updated_prompt;

        missing_images_created += 1;
    }
    let dest_file = fs::File::create(path)?;
    log.write_update(&dest_file)?;

    Ok(missing_images_created)
}

pub fn reroll_all(file_path: &str, api_url: Option<&str>) -> anyhow::Result<()> {
    // let mut log = BatchLog::from_file(file_path)?;
    let mut log = BatchLog::from_file(file_path)?;

    let path = PathBuf::from(file_path);
    let output_dir = path.parent().expect("couldn't get folder from file_path");

    let api = get_api_client(api_url, &None, &None)?;

    for (index, prompt) in log.images.clone().iter().enumerate() {
        print_reroll_start(index);

        let mut updated_prompt = prompt.to_owned();
        updated_prompt.seed = None;
        BatchTemplate::generate_image(output_dir, &api, &mut updated_prompt, index)?;
        log.images[index] = updated_prompt;
    }
    let dest_file = fs::File::create(path)?;
    log.write_update(&dest_file)?;

    Ok(())
}
