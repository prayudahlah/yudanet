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

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_save_load_roundtrip() {
        let path = "/tmp/test_roundtrip.yn";
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        t.save(path).unwrap();
        let loaded = Tensor::load(path).unwrap();
        assert_eq!(loaded.shape(), &[2, 3]);
        let orig: Vec<f32> = t.iter().collect();
        let loaded_vals: Vec<f32> = loaded.iter().collect();
        assert_eq!(orig, loaded_vals);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_load_strided() {
        let path = "/tmp/test_strided.yn";
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = t.slice(&[0..2, 0..1]);
        s.save(path).unwrap();
        let loaded = Tensor::load(path).unwrap();
        assert_eq!(loaded.shape(), &[2, 1]);
        let orig: Vec<f32> = t.slice(&[0..2, 0..1]).iter().collect();
        let loaded_vals: Vec<f32> = loaded.iter().collect();
        assert_eq!(orig, loaded_vals);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_invalid_magic() {
        use std::io::Write;
        let path = "/tmp/test_bad_magic.yn";
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"BAD\0\0\0\0\0").unwrap();
        f.write_all(&[1u8; 4]).unwrap();
        drop(f);
        assert!(Tensor::load(path).is_err());
        std::fs::remove_file(path).ok();
    }
}
