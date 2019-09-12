use tract_core::internal::*;
use tract_core::ops::cnn::*;

use crate::model::ParsingContext;
use crate::tfpb::node_def::NodeDef;

pub fn rates(pb: &NodeDef) -> TractResult<Vec<usize>> {
    let rates: Vec<usize> = pb.get_attr_list_int("rates")?;
    if rates.len() != 4 || rates[0] != 1 && rates[3] != 1 {
        Err(format!("rates must be of the form [1, h, v, 1], found {:?}", rates))?
    };
    Ok(rates)
}

pub fn ksizes(pb: &NodeDef) -> TractResult<Vec<usize>> {
    let ksizes: Vec<usize> = pb.get_attr_list_int("ksizes")?;
    if ksizes.len() != 4 || ksizes[0] != 1 && ksizes[3] != 1 {
        Err(format!("ksizes must be of the form [1, h, v, 1], found {:?}", ksizes))?
    };
    Ok(ksizes)
}


pub fn extract_image_patches(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let padding = super::padding(pb)?;
    let strides = super::strides(pb)?.into();
    let rates = rates(pb)?.into();
    let ksizes = ksizes(pb)?.into();
    Ok(Box::new(ExtractImagePatches { rates, strides, padding, ksizes }))
}


#[derive(Debug, Clone, new)]
struct ExtractImagePatches {
    rates: TVec<usize>,
    strides: TVec<usize>,
    padding: tract_core::ops::cnn::PaddingSpec,
    ksizes: TVec<usize>,
}

impl Op for FusedBatchNorm {
    fn name(&self) -> Cow<str> {
        "tf.FusedBatchNorm".into()
    }
}
