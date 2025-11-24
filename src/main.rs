mod args;
mod lstm;
use crate::args::CommandParse;
use crate::args::Commands;
use crate::lstm::kmeansnode;
use clap::Parser;
use tch::TchError;
use tch::Tensor::from_slice;
use tch::nn::{Adam, Module, Optimizer};
use tch::{Device, Kind, Tensor, nn, no_grad};

/*
Gaurav Sablok
codeprog@icloud.com
A week ago i put a post on how you can implement LSTM in RUST
and today i finished a part of coding a LSTM model in RUST.
I am only limited to the sequence and biological modelling and not audio and others.
*/

fn main() -> tch::TchResult<()> {
    let argparse = CommandParse::parse();
    match &argparse.command {
        Commands::LSTM {
            filepathinput,
            seqlen,
            batchsize,
            numepochs,
            hiddensize,
            numlayers,
            vocabsize,
        } => {
            let device = Device::Cpu;
            let vs = nn::VarStore::new(device);
            let net = Net::new(&vs, hiddensize, numlayers, vocabsize);
            let mut opt = Adam::default().build(&vs, 1e-3)?;
            let samples = load_real_data(filepathinput).expect("file not present");
            let num_samples = samples.len() as i64;
            let mut x_data: Vec<f32> = Vec::new();
            let mut y_data: Vec<f32> = Vec::new();
            for sample in samples {
                y_data.push(sample.label);
                for &idx in &sample.seq_indices {
                    let one_hot = match idx {
                        0 => vec![1.0, 0.0, 0.0, 0.0],
                        1 => vec![0.0, 1.0, 0.0, 0.0],
                        2 => vec![0.0, 0.0, 1.0, 0.0],
                        3 => vec![0.0, 0.0, 0.0, 1.0],
                        _ => vec![0.0, 0.0, 0.0, 0.0],
                    };
                    x_data.extend(one_hot);
                }
            }

            let x = from_slice(&x_data)
                .view([num_samples, seqlen, vocabsize])
                .to_kind(Kind::Float);
            let y = from_slice(&y_data)
                .view([num_samples, 1])
                .to_kind(Kind::Float);

            for epoch in 0..numepochs {
                let mut total_loss = 0.0;
                for start in (0..num_samples as usize).step_by(*batchsize as usize) {
                    let end = (start + *batchsize as usize).min(num_samples as usize);
                    let batch_size = (end - start) as i64;
                    if batch_size == 0 {
                        break;
                    }
                    let batch_x = x.narrow(0, start as i64, batch_size);
                    let batch_y = y.narrow(0, start as i64, batch_size);

                    let pred = net.forward(&batch_x);
                    let loss = nn::functional::binary_cross_entropy(
                        &pred,
                        &batch_y,
                        None::<Tensor>,
                        Kind::Float,
                    );
                    opt.backward_step(&loss);
                    total_loss += f64::from(loss.double());
                }
                if epoch % 10 == 0 {
                    println!(
                        "Epoch {}: Avg Loss = {:.4}",
                        epoch,
                        total_loss / (num_samples as f64 / *batchsize as f64)
                    );
                }
            }
            println!("The epochs for the lstm have finished: {:?}");
        }
    }
}
