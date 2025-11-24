use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use tch::Tensor::from_slice;
use tch::nn::{Adam, Module, Optimizer};
use tch::{Device, Kind, Tensor, nn, no_grad};

/*
Gaurav Sablok
codepro@icloud.com
*/

#[derive(Debug)]
struct DnaSample {
    label: f32,
    seq_indices: Vec<i64>,
}

/*
LSTM Model
*/

struct Net {
    lstm: nn::lstm::LSTM,
    linear: nn::Linear,
}

impl Net {
    fn new(vs: &nn::VarStore, hidden_size: i64, num_layers: i64, input_size: i64) -> Net {
        let lstm = nn::lstm::lstm(
            vs.root() / "lstm",
            input_size,
            hidden_size,
            nn::LSTMConfig::default().num_layers(num_layers),
        );
        let linear = nn::linear(vs.root() / "linear", hidden_size, 1, Default::default());
        Net { lstm, linear }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let (output, _) = self.lstm.seq(&xs.transpose(1, 2));
        let last_output = output.i((.., -1, ..)).squeeze_dim(1);
        self.linear.forward(&last_output).squeeze_dim(1)
    }
}

/*
 Loading the datasets
*/

fn load_real_data(filename: &str) -> Result<Vec<DnaSample>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b' ')
        .from_reader(BufReader::new(file));
    let mut samples = Vec::new();
    for result in rdr.records() {
        let record = result?;
        if record.len() != 49 {
            continue;
        }
        let class_str = record[0].trim();
        let label = if class_str == "+" { 1.0 } else { 0.0 };
        let mut seq_indices = Vec::new();
        for &nt_str in &record[2..] {
            let nt = nt_str
                .trim()
                .to_ascii_lowercase()
                .chars()
                .next()
                .unwrap_or('?');
            let idx = match nt {
                'a' => 0,
                'c' => 1,
                'g' => 2,
                't' => 3,
                _ => {
                    eprintln!("Unknown nucleotide: {}", nt);
                    continue;
                }
            };
            seq_indices.push(idx);
        }
        if seq_indices.len() == SEQ_LEN as usize {
            samples.push(DnaSample { label, seq_indices });
        }
    }
    println!(
        "Loaded {} real DNA samples from {}",
        samples.len(),
        filename
    );
    Ok(samples)
}
