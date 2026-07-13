use std::fmt::{Display, Formatter, Result};

use crate::Tensor;

impl Tensor {
    fn format(
        data: &[f32],
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        max_dim: usize,
    ) -> String {
        assert!(max_dim >= strides.len());

        let mut formatted: String = String::new();

        if max_dim == strides.len() {
            formatted.push_str("Tensor");
        }

        if shape.len() == 1 {
            formatted.push('[');

            for i in 0..shape[0] {
                let idx = offset + strides[0] * i;

                if i > 0 {
                    formatted.push_str(", ");
                }

                formatted.push_str(&format!("{:.2}", data[idx]));
            }

            formatted.push(']');
        } else {
            formatted.push('[');

            for i in 0..shape[0] {
                let sub_offset = offset + strides[0] * i;

                formatted.push_str(&Self::format(
                    data,
                    &shape[1..shape.len()],
                    &strides[1..strides.len()],
                    sub_offset,
                    max_dim,
                ));

                if i < shape[0] - 1 {
                    formatted.push_str(&format!(
                        ",{}{}",
                        "\n".repeat((shape.len() - 1).max(1)),
                        " ".repeat(6 + max_dim - shape.len() + 1)
                    ));
                }
            }

            formatted.push(']');
        }

        formatted
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let guard = self.data();
        let data: &[f32] = guard.as_slice();

        write!(
            f,
            "{}",
            Self::format(
                data,
                self.shape(),
                self.strides(),
                self.offset(),
                self.strides().len()
            )
        )
    }
}
