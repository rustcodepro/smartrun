use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "smartrun",
    version = "1.0",
    about = "Fitting LSTM layer in RUST to sequence data.
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// LSTM
    LSTM {
        /// file path of the fasta
        filepathinput: String,
        /// seqlen for the segmentation
        seqlen: i64,
        /// batch size
        batchsize: i64,
        /// epochs number
        numepochs: i64,
        /// hidden size
        hiddensize: i64,
        /// number of the layers
        numlayers: i64,
        /// vocabsize
        vocabsize: i64,
    },
}
