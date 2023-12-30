use base64::{
    alphabet,
    engine::{self, general_purpose},
    Engine as _,
};

use reqwest::blocking::ClientBuilder;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::BatchError;

#[derive(Serialize, Deserialize)]
struct Sampler {
    name: String,
    aliases: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct SDModel {
    title: String,
    model_name: String,
    hash: Option<String>,
    sha256: Option<String>,
    filename: String,
    config: Option<String>,
}

#[derive(Deserialize)]
struct Upscaler {
    name: String,
    scale: f32,
}

/// Many more options are available, but we only care about these
#[derive(Serialize, Deserialize)]
struct SDAPIOptions {
    sd_model_checkpoint: String,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
struct PromptData {
    /// Positive prompt
    prompt: String,
    negative_prompt: String,
    /// Sampler to use, ex., "DPM 2M++ Karras"
    sampler_name: String,
    steps: u32,
    /// Image width at generation time, before Hi-res
    width: u32,
    /// Image height at generation time, before Hi-res
    height: u32,
    cfg_scale: f32,
    seed: i64,
    enable_hr: bool,
    hr_scale: f32,
    hr_upscaler: String,
    hr_second_pass_steps: u8,
    send_images: bool,
    save_images: bool,
    restore_faces: bool,
}

impl From<&super::PromptData> for PromptData {
    fn from(value: &super::PromptData) -> Self {
        PromptData {
            prompt: value.positive.clone(),
            negative_prompt: value.negative.clone(),
            sampler_name: value.sampler.clone(),
            steps: value.steps,
            width: value.width,
            height: value.height,
            cfg_scale: value.cfg,
            seed: value.seed.unwrap_or(-1),
            enable_hr: value.hires.is_some(),
            hr_scale: match &value.hires {
                Some(hires) => hires.upscale_by,
                None => 1.0,
            },
            hr_upscaler: match &value.hires {
                Some(hires) => hires.upscaler.clone(),
                None => "".to_string(),
            },
            hr_second_pass_steps: match &value.hires {
                Some(hires) => hires.steps,
                None => 0,
            },
            send_images: true,
            save_images: false,
            restore_faces: false,
        }
    }
}

/*
Automatic1111 API notes
GET /sdapi/v1/progress <- might be cool to use this to show a progress bar in the terminal?
*/

#[derive(Serialize, Deserialize, Debug)]
struct Txt2ImgResponse {
    images: Vec<String>,
    parameters: PromptData, // not sure what this type is
    info: String,
}

pub struct APIClient {
    api_url: String,
    client: reqwest::blocking::Client,
}

impl APIClient {
    pub fn new(api_url: &str) -> anyhow::Result<APIClient> {
        let timeout = std::time::Duration::new(90, 0);
        let client = ClientBuilder::new().timeout(timeout).build()?;

        Ok(APIClient {
            api_url: api_url.to_owned(),
            client,
        })
    }

    fn get_samplers(&self) -> anyhow::Result<Vec<Sampler>> {
        let resp = self
            .client
            .get(format!("{}/sdapi/v1/samplers", &self.api_url))
            .send()?;
        let samplers: Vec<Sampler> = resp.json()?;
        Ok(samplers)
    }

    fn invalid_sampler(&self, prompt: &PromptData) -> anyhow::Result<BatchError> {
        let samplers = self.get_samplers()?;
        let mut error_msg = format!(
            "Sampler \"{}\" not found, must be one of: \n",
            prompt.sampler_name
        );
        for sampler in samplers {
            error_msg.push_str(&format!("{}\n", sampler.name));
            for alias in sampler.aliases {
                error_msg.push_str(&format!(" alias: {}\n", alias));
            }
        }
        Ok(super::BatchError { message: error_msg })
    }

    fn get_checkpoints(&self) -> anyhow::Result<Vec<SDModel>> {
        let resp = self
            .client
            .get(format!("{}/sdapi/v1/sd-models", &self.api_url))
            .send()?;
        let samplers: Vec<SDModel> = resp.json()?;
        Ok(samplers)
    }

    fn invalid_model(&self, model: &str) -> anyhow::Result<BatchError> {
        let models = self.get_checkpoints()?;
        let mut error_msg = format!("Model \"{}\" not found, must be one of: \n", model);
        for model in models {
            error_msg.push_str(&format!("{}\n", model.title));
            error_msg.push_str(&format!("  ({})\n", model.model_name));
        }
        Ok(super::BatchError { message: error_msg })
    }

    fn get_upscalers(&self) -> anyhow::Result<Vec<Upscaler>> {
        let resp = self
            .client
            .get(format!("{}/sdapi/v1/upscalers", &self.api_url))
            .send()?;
        let upscalers: Vec<Upscaler> = resp.json()?;
        Ok(upscalers)
    }

    fn invalid_upscaler(&self, upscaler: &str) -> anyhow::Result<BatchError> {
        let upscalers = self.get_upscalers()?;
        let mut error_msg = format!("Upscaler \"{}\" not found, must be one of: \n", upscaler);
        for upscaler in upscalers {
            error_msg.push_str(&format!("{}\n", upscaler.name));
        }
        Ok(super::BatchError { message: error_msg })
    }

    pub fn txt2img(&self, prompt: &super::PromptData) -> anyhow::Result<(Vec<Vec<u8>>, String)> {
        self.ensure_model(&prompt.model)?;

        let prompt: PromptData = prompt.into();

        let resp = self
            .client
            .post(format!("{}/sdapi/v1/txt2img", &self.api_url))
            .json(&prompt)
            .send()?;
        let resp_status = resp.status();
        if resp_status != StatusCode::OK {
            let error_msg = &resp.text()?;

            if error_msg.contains("Sampler not found") {
                let sampler_error = self.invalid_sampler(&prompt)?;
                return Err(sampler_error.into());
            }
            if error_msg.contains("could not find upscaler named") {
                let upscaler_error = self.invalid_upscaler(&prompt.hr_upscaler)?;
                return Err(upscaler_error.into());
            }
            return Err(super::BatchError {
                message: format!(
                    "Unexpected response when trying txt2img: {}, {:?}",
                    resp_status, error_msg
                ),
            }
            .into());
        }

        let resp: Txt2ImgResponse = resp.json()?;
        let mut image_list = vec![];
        for base64image in resp.images {
            let image_bytes = general_purpose::STANDARD.decode(base64image).unwrap();
            image_list.push(image_bytes);
        }

        Ok((image_list, resp.info))
    }

    fn ensure_model(&self, model: &str) -> anyhow::Result<()> {
        let resp: SDAPIOptions = self
            .client
            .get(format!("{}/sdapi/v1/options", &self.api_url))
            .send()?
            .json()?;
        if resp.sd_model_checkpoint != model {
            let resp = self
                .client
                .post(format!("{}/sdapi/v1/options", &self.api_url))
                .json(&SDAPIOptions {
                    sd_model_checkpoint: model.to_string(),
                })
                .send()?;
            let resp_status = resp.status();
            if resp_status != StatusCode::OK {
                let error_msg = resp.text()?;
                if error_msg.contains(&format!("model '{}' not found", model)) {
                    let model_error = self.invalid_model(&model)?;
                    return Err(model_error.into());
                }
                return Err(super::BatchError {
                    message: format!(
                        "Unexpected response when trying to set model option: {}, {:?}",
                        resp_status, error_msg
                    ),
                }
                .into());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn test_decode_txt2image_response() {
        let mock_response = fs::read_to_string("testdata/debug-response-2.json")
            .expect("error reading from testdata");

        let _response: Txt2ImgResponse =
            serde_json::from_str(mock_response.as_str()).expect("error deserializing json");
    }
}
