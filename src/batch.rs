use chrono::Local;
use rand::seq::SliceRandom;
use rand::Rng;
use std::{
    fmt::{self},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use self::auto1111_api::APIClient;

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
    seed: Option<i64>,
    hires: Option<HiResSettings>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct HiResSettings {
    upscaler: String,
    upscale_by: f32,
    denoising_strength: f32,
    steps: u8,
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

    /// The pool of prompts to pick from
    pub prompts: Vec<Prompts>,

    /// Additional modifiers to add to each prompt
    pub modifiers: Option<Vec<WeightedPrompt>>,
}

fn get_api_client(api_url: Option<&str>) -> anyhow::Result<APIClient> {
    let url_to_use = match api_url {
        Some(url) => url,
        None => "http://127.0.0.1:7860",
    };
    println!("Using API at: {}", url_to_use);
    let api = auto1111_api::APIClient::new(&url_to_use)?;
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
    fn run(&self, dry_run: bool, output_dir: &Path, sequential: bool, api_url: Option<&str>) -> anyhow::Result<BatchLog> {
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

        let mut batch_log = BatchLog::new(&self.name);

        if !dry_run {
            std::fs::create_dir_all(output_dir)?;
        }

        let api = if dry_run {
            None
        } else {
            Some(get_api_client(api_url)?)
        };

        for (prompt_index, prompt) in prompt_pool.iter().enumerate() {
            let prompt_data = self.run_prompt(&api, output_dir, prompt, prompt_index)?;
            batch_log.images.push(prompt_data);
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

    fn run_prompt(
        &self,
        api: &Option<APIClient>,
        output_dir: &Path,
        prompt: &Prompts,
        prompt_index: usize,
    ) -> anyhow::Result<PromptData> {
        let mut rng = rand::thread_rng();
        let mut prompt_data = match prompt {
            Prompts::Single(positive) => self.copy_with_positive(positive),
            Prompts::Multiple(positive_vec) => {
                let positive = positive_vec.choose(&mut rng);
                self.copy_with_positive(
                    positive.expect("Prompts::Multiple to always pick Some positive prompt"),
                )
            }
        };

        if let Some(modifiers) = &self.modifiers {
            if let Some(modifier) = modifiers.choose(&mut rng) {
                let roll: f32 = rng.gen();
                if roll <= modifier.chance {
                    // ring-a-ding-ding!
                    prompt_data.positive =
                        Self::combine_prompts(&prompt_data.positive, &modifier.prompt);
                }
            }
        }

        if let Some(api) = api {
            println!("Generating image {}...", prompt_index + 1);
            Self::generate_image(output_dir, &api, &mut prompt_data, prompt_index)?;
        }

        Ok(prompt_data)
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

            let mut dest_file = fs::File::create(&image_filename)?;
            dest_file.write_all(image_bytes)?;
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
    // Idea: MultipleWeighted(Vec<WeightedPrompt>), struct option similar to Multiple but with a weight specified for each prompt
}

#[derive(Serialize, Deserialize)]
pub struct WeightedPrompt {
    /// Prompt string to use
    prompt: String,
    /// The % chance for this prompt to be picked
    ///
    /// ex., "0.8" would be an 80% chance
    chance: f32,
}

#[derive(Serialize, Deserialize)]
pub struct BatchLog {
    /// Template name used for generation
    template: String,

    /// Generated images
    images: Vec<PromptData>,
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
    fn new(name: &str) -> BatchLog {
        return BatchLog {
            template: name.to_owned(),
            images: vec![],
        };
    }

    fn from_file(file_path: &str) -> anyhow::Result<BatchLog> {
        let log_file = fs::File::open(file_path)?;
        let log: BatchLog = serde_json::from_reader(log_file)?;
        Ok(log)
    }

    fn safe_logfile_name(&self, name: &str) -> PathBuf {
        let timestamp = Local::now();
        let timestamp = format!("{}", timestamp.format("%Y-%m-%d-%H%M"));

        let sanitized_filename = get_safe_filename(name);
        return format!("{}-{}.json", sanitized_filename, timestamp).into();
    }

    /// Serialize to JSON and write to disk
    pub fn write(&self, output_dir: &Path, template_name: &str) -> anyhow::Result<PathBuf> {
        let dest_filename = self.safe_logfile_name(template_name);
        let (_, dest_file) = create_file_and_dir(output_dir, &dest_filename)?;
        serde_json::to_writer_pretty(&dest_file, self)?;

        Ok(dest_filename)
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
    api_url: Option<&str>
) -> anyhow::Result<TemplateRunResults> {
    let json_string = fs::read_to_string(template_filename)?;
    let template: BatchTemplate = serde_json::from_str(&json_string)?;

    let output_dir = PathBuf::from(output_dir);

    let batch_log = template.run(dry_run, &output_dir, sequential, api_url)?;
    let log_file = batch_log.write(&output_dir, &template.name)?;

    Ok(TemplateRunResults {
        images_created: batch_log.images.len(),
        log_file,
    })
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
            let api = get_api_client(api_url)?;

            let mut updated_prompt = prompt.to_owned();
            updated_prompt.seed = None;
            BatchTemplate::generate_image(output_dir, &api, &mut updated_prompt, index)?; 
            log.images[index] = updated_prompt;
            let dest_file = fs::File::create(path)?;
            log.write_update(&dest_file)?;

            Ok(())
        }
    }
}

pub fn reroll_all(file: &str) {
    todo!()
    // let mut log = BatchLog::from_file(file_path)?;
}
