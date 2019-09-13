use tract_core::internal::*;

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

pub fn extract_image_patches(
    _ctx: &ParsingContext,
    pb: &NodeDef,
) -> TractResult<Box<dyn InferenceOp>> {
    let padding = super::padding(pb)?;
    let strides = super::strides(pb)?[1..3].into();
    let rates = rates(pb)?[1..3].into();
    let ksizes = ksizes(pb)?[1..3].into();
    Ok(Box::new(ExtractImagePatches { rates, strides, padding, ksizes }))
}

#[derive(Debug, Clone, new)]
struct ExtractImagePatches {
    rates: TVec<usize>,
    strides: TVec<usize>,
    padding: tract_core::ops::cnn::PaddingSpec,
    ksizes: TVec<usize>,
}

impl ExtractImagePatches {}

impl Op for ExtractImagePatches {
    fn name(&self) -> Cow<str> {
        "tf.ExtractImagePatches".into()
    }

    op_as_typed_op!();
}

impl StatelessOp for ExtractImagePatches {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unimplemented!()
    }
}

impl InferenceRulesOp for ExtractImagePatches {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;

        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&outputs[0].rank, 4)?;
        s.equals(&inputs[0].shape[0], &outputs[0].shape[0])?;
        s.equals(
            &outputs[0].shape[3],
            (self.ksizes[0] * self.ksizes[1]) as i32 * inputs[0].shape[3].bex(),
        )?;
        s.given(&inputs[0].shape, move |s, shape| {
            let computed =
                self.padding.compute(&shape[1..3], &*self.ksizes, &*self.rates, &*self.strides);
            s.equals(&outputs[0].shape[1], &computed[0].output)?;
            s.equals(&outputs[0].shape[2], &computed[1].output)?;
            Ok(())
        })?;
        Ok(())
    }
    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for ExtractImagePatches {
    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        let input_shape = inputs[0].shape.to_tvec();
        let computed =
            self.padding.compute(&input_shape[1..3], &*self.ksizes, &*self.rates, &*self.strides);
        let output_shape = [
            input_shape[0].clone(),
            computed[0].output.clone(),
            computed[1].output.clone(),
            input_shape[3].clone() * (self.ksizes[0] * self.ksizes[1]) as i32,
        ];
        Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, output_shape.as_ref())?))
    }
    typed_op_as_op!();
}
