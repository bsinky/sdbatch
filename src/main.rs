use std::path;

use batch::BatchTemplate;
use clap::{Parser, Subcommand};

mod batch;

fn main() {
    let args = Args::parse();
    
    match args.command {
        None => {
            println!("Nothing to do, you need to use a command!")
        },
        Some(command) => {
            match command {
                Commands::Run { dry_run, file, output } => {
                    match batch::do_run(dry_run, &file, &output) {
                        Ok(results) => {
                            println!("Template run successful, created {} images", results.images_created)
                        },
                        Err(err) => {
                            println!("Template run error: {:?}", err)
                        }
                    }
                }
                Commands::Reroll { file, index, all } => {
                    if index.is_none() && !all {
                        println!("Reroll requires either --all or an INDEX to run")
                    }
                    else {
                        todo!()
                    }
                },
                Commands::Create { name, output_dir } => {
                    let mut template:BatchTemplate = Default::default();
                    template.name = name;
                    let output_dir = &output_dir.unwrap_or("./".to_string());
                    let output_path = path::Path::new(output_dir);
                    match template.write(output_path) {
                        Ok(dest_filename) => println!("Created blank template file: {}", dest_filename.display()),
                        Err(err) => {
                            println!("Template generation error: {:?}", err)
                        }
                    }
                },
            }
        }
    }
}

#[derive(Parser)]
#[command(name = "SD Batch")]
#[command(author = "bsinky")]
#[command(version = "0.1")]
#[command(about = "Batch generate images using Automatic1111 from a given template file", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Run {
        /// Do not generate images, merely process input files and generate prompts
        #[arg(short = 'n', long)]
        dry_run: bool,

        /// JSON input file for batch template
        file: String,

        /// Output directory to save log and images
        output: String,
    },
    Reroll {
        /// Regenerate all images from the log
        #[arg(long)]
        all: bool,

        /// Batch log file to reroll for
        file: String,

        /// Image index to reroll
        index: Option<usize>,
    },
    /// Generate an empty Template file
    Create {
        /// Name of the blank Template to generate
        name: String,

        /// Directory to place the generated Template in, defaults to current directory
        #[arg(short, long)]
        output_dir: Option<String>,
    }
}
