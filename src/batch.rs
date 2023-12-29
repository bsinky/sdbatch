use chrono::Local;
use rand::seq::SliceRandom;
use rand::Rng;
use std::{
    fmt::{self},
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct PromptData {
    positive: String,
    negative: String,
    model: String,
    /// Sampler to use, ex., "DPM 2M++"
    sampler: String,
    steps: u32,
    /// Image resolution at generation time, before Hi-res
    resolution: String,
    cfg: f32,
    seed: Option<u64>,
    hires: Option<HiResSettings>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct HiResSettings {
    upscaler: String,
    upscale_by: u8,
    denoising_strength: f32,
}

#[derive(Serialize, Deserialize, Default)]
pub struct BatchTemplate {
    /// Short name, used in the filename for logs after template runs
    pub name: String,

    /// Longer, detailed description
    pub description: Option<String>,

    /// Automatic1111 URL
    ///
    /// Defaults to 127.0.0.1:7690
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

impl BatchTemplate {
    fn run(&self, dry_run: bool, sequential: bool) -> anyhow::Result<BatchLog> {
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

        for prompt in prompt_pool {
            let prompt_data = self.run_prompt(dry_run, prompt)?;
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

    fn run_prompt(&self, dry_run: bool, prompt: &Prompts) -> anyhow::Result<PromptData> {
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

        if !dry_run {
            self.generate_image(&mut prompt_data)?;
        }

        Ok(prompt_data)
    }

    /// Use the Automatic1111 API and generate an image for the given prompt, and sets the seed
    fn generate_image(&self, prompt: &mut PromptData) -> anyhow::Result<()> {
        todo!()
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
    // TODO: MultipleWeighted(Vec<WeightedPrompt>), struct option similar to Multiple but with a weight specified for each prompt
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
}

pub struct TemplateRunResults {
    pub images_created: usize,
    pub log_file: PathBuf,
}

pub fn do_run(
    dry_run: bool,
    sequential: bool,
    template_filename: &str,
    output_filename: &str,
) -> anyhow::Result<TemplateRunResults> {
    let json_string = fs::read_to_string(template_filename)?;
    let template: BatchTemplate = serde_json::from_str(&json_string)?;

    let batch_log = template.run(dry_run, sequential)?;
    let log_file = batch_log.write(Path::new(output_filename), &template.name)?;

    Ok(TemplateRunResults {
        images_created: batch_log.images.len(),
        log_file,
    })
}

pub fn reroll(file: &str, index: usize) {
    todo!()
}
