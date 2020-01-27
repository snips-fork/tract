use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct AddDims {
    pub axes: Vec<usize>,
}

impl AddDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut shape: TVec<D> = input.iter().cloned().collect();
        for &axis in &self.axes {
            shape.insert(axis, D::one())
        }
        shape
    }
}

impl Op for AddDims {
    fn name(&self) -> Cow<str> {
        "AddDims".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axes: {:?}", self.axes)])
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for AddDims {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let shape = self.compute_shape(input.shape());
        Ok(unsafe { tvec![input.into_tensor().into_shape(&*shape)?.into_arc_tensor()] })
    }
}

impl InferenceRulesOp for AddDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() + self.axes.len() as i32)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for AddDims {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            self.compute_shape(&*inputs[0].shape.to_tvec()).as_ref(),
        )?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.axes.len() == 0 {
            Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
        } else {
            Ok(None)
        }
    }

    fn invariants(&self, _model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let mut i = 0;
        let mut axes = tvec!();
        for out in 0..node.outputs[0].fact.shape.rank() {
            if !self.axes.contains(&out) {
                axes.push(AxisInfo {
                    inputs: tvec!(Some(i)),
                    outputs: tvec!(Some(out)),
                    period: 1,
                    disposable: true,
                });
                i += 1;
            }
        }
        Ok(axes.into_iter().collect())
    }

    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        Ok(self
            .axes
            .iter()
            .enumerate()
            .map(|(ix, axis)| (InOut::Out(ix), AxisOp::Rm(*axis)))
            .collect())
    }

    fn change_axes(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        fn rm_axis(axes: &[usize], axis: usize) -> Vec<usize> {
            axes.iter().filter(|&a| a != &axis).map(|&a| a - (a > axis) as usize).collect()
        }
        match change {
            AxisOp::Rm(axis) => match io {
                InOut::Out(_) if self.axes.contains(&axis) => Ok(Some(AxisChangeConsequence {
                    substitute_op: Some(Box::new(AddDims::new(rm_axis(&self.axes, *axis)))),
                    wire_changes: tvec!(),
                })),
                InOut::Out(_) => Ok(Some(AxisChangeConsequence {
                    substitute_op: Some(Box::new(AddDims::new(rm_axis(&self.axes, *axis)))),
                    wire_changes: tvec!((
                        InOut::In(0),
                        AxisOp::Rm(axis - self.axes.iter().filter(|&x| x < axis).count())
                    )),
                })),
                InOut::In(_) => Ok(Some(AxisChangeConsequence {
                    substitute_op: Some(Box::new(AddDims::new(rm_axis(&self.axes, *axis)))),
                    wire_changes: tvec!((
                        InOut::Out(0),
                        AxisOp::Rm(axis + self.axes.iter().filter(|&x| x < axis).count())
                    )),
                })),
            },
        }
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        target.wire_node(&*node.name, self.clone(), &[input])
    }
}

impl PulsedOp for AddDims {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape = self.compute_shape(&*inputs[0].shape);
        fact.axis += self.axes.iter().filter(|&ax| *ax <= fact.axis).count();
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
