use std::{
    fs::File,
    io::{BufReader, BufWriter, Error, Read, Result, Write},
};

use crate::Tensor;

impl Tensor {
    pub fn save(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header (Magic Bytes + ndim)
        writer.write_all(b"yn\0\0\0\0\0\0")?;
        writer.write_all(&(self.ndim() as u32).to_le_bytes())?;

        // Shape
        self.shape()
            .iter()
            .try_for_each(|&x| writer.write_all(&(x as u64).to_le_bytes()))?;

        // Data
        self.iter()
            .try_for_each(|x| writer.write_all(&x.to_le_bytes()))?;

        writer.flush()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Tensor> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Header (Magic Bytes + ndim)
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;

        if &magic != b"yn\0\0\0\0\0\0" {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        let mut ndim_buf = [0u8; 4];
        reader.read_exact(&mut ndim_buf)?;
        let ndim = u32::from_le_bytes(ndim_buf) as usize;

        // Shape
        let mut shape: Vec<usize> = vec![0; ndim];
        for i in 0..ndim {
            let mut shape_buf = [0u8; 8];
            reader.read_exact(&mut shape_buf)?;
            shape[i] = u64::from_le_bytes(shape_buf) as usize;
        }

        // Data
        let length = shape.iter().product();
        let mut data: Vec<f32> = vec![0.0; length];
        for i in 0..length {
            let mut data_buf = [0u8; 4];
            reader.read_exact(&mut data_buf)?;
            data[i] = f32::from_le_bytes(data_buf);
        }

        Ok(Self::new(data, shape))
    }
}
