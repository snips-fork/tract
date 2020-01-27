use num_traits::Zero;
use std::fmt;
use std::ops::{Add, Mul};

use crate::internal::*;
use crate::ops::matmul::*;
use crate::ops::quant::QParams;
use ndarray::*;

fn eval(
    a: &Tensor,
    b: &Tensor,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    q_params: Option<&QParams>,
) -> TractResult<Tensor> {
    if let Some(q) = q_params {
        if (a.datum_type(), b.datum_type()) == (i8::datum_type(), i8::datum_type()) {
            return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i32)(m, k, n))
            });
        } else if (a.datum_type(), b.datum_type()) == (i8::datum_type(), i8::datum_type()) {
            if q.c_datum_type == i32::datum_type() {
                return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i32)(m, k, n))
                });
            } else if q.c_datum_type == i8::datum_type() {
                return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i8)(m, k, n))
                });
            }
        } else if (a.datum_type(), b.datum_type()) == (u8::datum_type(), u8::datum_type()) {
            if q.c_datum_type == i32::datum_type() {
                return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_u8_i32)(m, k, n))
                });
            } else if q.c_datum_type == u8::datum_type() {
                return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_u8_u8)(m, k, n))
                });
            }
        }
    } else if (a.datum_type(), b.datum_type()) == (f32::datum_type(), f32::datum_type()) {
        return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
            MMMWrapper::Plain((tract_linalg::ops().smmm)(m, k, n))
        });
    }
    bail!(
        "Unsupported combination for MatMul eval (a: {:?}, b:{:?} q:{:?})",
        a.datum_type(),
        b.datum_type(),
        q_params
    );
}

fn eval_t<TA, TB, TC, TI>(
    a: &Tensor,
    b: &Tensor,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    q_params: Option<&QParams>,
    mmm: impl Fn(usize, usize, usize) -> MMMWrapper<TA, TB, TC, TI>,
) -> TractResult<Tensor>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy + Zero + fmt::Debug,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    let a = a.to_array_view::<TA>()?;
    let b = b.to_array_view::<TB>()?;
    let mut geo = Geo::<TA, TB, TC, TI>::new(a.shape(), b.shape(), a_trans, b_trans, c_trans, mmm)?;
    unsafe {
        geo.mm.as_mmm_mut().c_from_data_and_strides(
            if c_trans { 1 } else { *geo.bc_c_shape.last().unwrap() as isize },
            if !c_trans { 1 } else { *geo.bc_c_shape.last().unwrap() as isize },
        );
        if let Some(q) = q_params {
            geo.mm.set_quant_params(q)?;
        }
    }
    let a = a.into_shape(&*geo.bc_a_shape)?;
    let b = b.into_shape(&*geo.bc_b_shape)?;
    let mut c = unsafe { Array::<TC, IxDyn>::uninitialized(&*geo.bc_c_shape) };

    let b_pack = geo.mm.as_mmm().b_pack();

    let mut pa = unsafe {
        Tensor::uninitialized_aligned::<TA>(
            &[geo.mm.as_mmm().a_pack().len()],
            geo.mm.as_mmm().a_pack().alignment(),
        )?
    };
    let mut pb =
        unsafe { Tensor::uninitialized_aligned::<TB>(&[b_pack.len()], b_pack.alignment())? };

    for prefix in indices(&*geo.c_shape_prefix).into_iter() {
        let mut a = a.view();
        let mut b = b.view();
        let mut c = c.view_mut();
        for (axis, &dim) in prefix.slice().iter().enumerate() {
            let d = dim.min(a.shape()[axis] - 1);
            a.slice_axis_inplace(Axis(axis), (d..=d).into());
            let d = dim.min(b.shape()[axis] - 1);
            b.slice_axis_inplace(Axis(axis), (d..=d).into());
            c.slice_axis_inplace(Axis(axis), (dim..=dim).into());
        }
        geo.mm.as_mmm().a_pack().pack(
            pa.as_ptr_mut()?,
            a.as_ptr(),
            a.strides()[prefix.ndim() + a_trans as usize],
            a.strides()[prefix.ndim() + !a_trans as usize],
        );
        b_pack.pack(
            pb.as_ptr_mut()?,
            b.as_ptr(),
            b.strides()[prefix.ndim() + b_trans as usize],
            b.strides()[prefix.ndim() + !b_trans as usize],
        );
        unsafe {
            geo.mm.run(pa.as_ptr()?, pb.as_ptr()?, c.as_mut_ptr(), &[]);
        }
    }
    unsafe { Ok(c.into_tensor().into_shape(&geo.final_c_shape)?) }
}

pub fn infer_shapes<D: DimLike>(
    ashape_orig: TVec<D>,
    bshape_orig: TVec<D>,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
) -> TractResult<(TVec<D>, TVec<D>, TVec<D>, TVec<D>)> {
    let mut ashape = ashape_orig.clone();
    let mut bshape = bshape_orig.clone();
    let mut implicit_m = false;
    let mut implicit_n = false;
    if ashape.len() < 2 {
        implicit_m = true;
        ashape.insert(a_trans as usize, D::one());
    }
    if bshape.len() < 2 {
        implicit_n = true;
        bshape.insert(!b_trans as usize, D::one());
    }
    while ashape.len() < bshape.len() {
        ashape.insert(0, D::one());
    }
    while bshape.len() < ashape.len() {
        bshape.insert(0, D::one());
    }
    let c_bc_shape_prefix = crate::broadcast::multi_broadcast(&[
        &ashape[..(ashape.len() - 2)],
        &bshape[..(bshape.len() - 2)],
    ])
    .ok_or("Could not broadcast")?;
    let mut c_bc_shape: TVec<D> = c_bc_shape_prefix.clone();
    let (mut m, mut ka) = (ashape[ashape.len() - 2].clone(), ashape[ashape.len() - 1].clone());
    let (mut kb, mut n) = (bshape[bshape.len() - 2].clone(), bshape[bshape.len() - 1].clone());
    if a_trans {
        std::mem::swap(&mut m, &mut ka);
    }
    if b_trans {
        std::mem::swap(&mut kb, &mut n);
    }
    if ka != kb {
        bail!(
            "Inconsistent malmul: a: {:?} b: {:?}, a_trans: {} b_trans: {} c_trans: {}",
            ashape,
            bshape,
            a_trans,
            b_trans,
            c_trans
        );
    }
    let mut c_shape_final = c_bc_shape.clone();
    if c_trans {
        c_bc_shape.push(n.clone());
        c_bc_shape.push(m.clone());
        if !implicit_n {
            c_shape_final.push(n.clone());
        }
        if !implicit_m {
            c_shape_final.push(m.clone());
        }
    } else {
        c_bc_shape.push(m.clone());
        c_bc_shape.push(n.clone());
        if !implicit_m {
            c_shape_final.push(m.clone());
        }
        if !implicit_n {
            c_shape_final.push(n.clone());
        }
    }
    Ok((ashape, bshape, c_bc_shape, c_shape_final))
}

#[derive(Debug, Clone)]
struct Geo<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    m: usize,
    k: usize,
    n: usize,
    mm: MMMWrapper<TA, TB, TC, TI>,
    a_shape: TVec<usize>,
    a_trans: bool,
    b_shape: TVec<usize>,
    b_trans: bool,
    bc_a_shape: TVec<usize>,
    bc_b_shape: TVec<usize>,
    bc_c_shape: TVec<usize>,
    final_c_shape: TVec<usize>,
    c_trans: bool,
    c_shape_prefix: TVec<usize>,
    a_stride_prefix: TVec<usize>,
    b_stride_prefix: TVec<usize>,
    c_stride_prefix: TVec<usize>,
}

impl<TA, TB, TC, TI> Geo<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    pub fn new(
        a_shape: &[usize],
        b_shape: &[usize],
        a_trans: bool,
        b_trans: bool,
        c_trans: bool,
        mmm: impl Fn(usize, usize, usize) -> MMMWrapper<TA, TB, TC, TI>,
    ) -> TractResult<Geo<TA, TB, TC, TI>> {
        let (bc_a_shape, bc_b_shape, bc_c_shape, final_c_shape) =
            infer_shapes(a_shape.into(), b_shape.into(), a_trans, b_trans, c_trans)?;
        let m = bc_a_shape[bc_a_shape.len() - 2 + a_trans as usize];
        let k = bc_a_shape[bc_a_shape.len() - 1 - a_trans as usize];
        let n = bc_b_shape[bc_b_shape.len() - 1 - b_trans as usize];
        let mm = mmm(m, k, n);
        let a_stride_prefix = bc_a_shape
            .iter()
            .rev()
            .scan(1, |stride, dim| {
                let s = Some(*stride);
                *stride *= dim;
                s
            })
            .skip(2)
            .collect();
        let b_stride_prefix = bc_b_shape
            .iter()
            .rev()
            .scan(1, |stride, dim| {
                let s = Some(*stride);
                *stride *= dim;
                s
            })
            .skip(2)
            .collect();
        let c_stride_prefix = bc_c_shape
            .iter()
            .rev()
            .scan(1, |stride, dim| {
                let s = Some(*stride);
                *stride *= dim;
                s
            })
            .skip(2)
            .collect();
        Ok(Geo {
            m,
            k,
            n,
            mm,
            c_shape_prefix: bc_c_shape[0..(bc_c_shape.len().saturating_sub(2))].into(),
            bc_a_shape,
            bc_b_shape,
            bc_c_shape,
            final_c_shape,
            a_shape: a_shape.into(),
            b_shape: b_shape.into(),
            a_stride_prefix,
            b_stride_prefix,
            c_stride_prefix,
            a_trans,
            b_trans,
            c_trans,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct MatMul {
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    q_params: Option<QParams>,
}

impl MatMul {
    pub fn with_a_trans(self, a_trans: bool) -> MatMul {
        MatMul { a_trans, ..self }
    }

    pub fn with_b_trans(self, b_trans: bool) -> MatMul {
        MatMul { b_trans, ..self }
    }

    pub fn with_c_trans(self, c_trans: bool) -> MatMul {
        MatMul { c_trans, ..self }
    }

    pub fn with_q_params(self, q_params: QParams) -> MatMul {
        MatMul { q_params: Some(q_params), ..self }
    }
}

impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for MatMul {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let t = eval(
            &inputs[0],
            &inputs[1],
            self.a_trans,
            self.b_trans,
            self.c_trans,
            self.q_params.as_ref(),
        )?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl InferenceRulesOp for MatMul {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        if let Some(qp) = &self.q_params {
            s.equals(&outputs[0].datum_type, &qp.c_datum_type)?;
        } else {
            s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        }
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, ashape, bshape| {
            let (_, _, _, cshape) =
                infer_shapes(ashape, bshape, self.a_trans, self.b_trans, self.c_trans)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for MatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = self.q_params.as_ref().map(|qp| qp.c_datum_type).unwrap_or(inputs[0].datum_type);
        Ok(tvec!(TypedFact::dt_shape(
            dt,
            &*infer_shapes(
                inputs[0].shape.to_tvec(),
                inputs[1].shape.to_tvec(),
                self.a_trans,
                self.b_trans,
                self.c_trans,
            )?
            .3
        )?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;
        let konst_ix = if a_fact.konst.is_some() {
            0
        } else if b_fact.konst.is_some() {
            1
        } else {
            return Ok(None);
        };

        let var_ix = 1 - konst_ix;
        let flip = konst_ix == 1;
        let t_konst = [self.a_trans, self.b_trans][konst_ix] ^ flip;
        let t_var = [self.b_trans, self.a_trans][konst_ix] ^ flip;
        let konst = model.outlet_fact(node.inputs[konst_ix])?.konst.clone().unwrap();
        let patch = TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs[var_ix..][..1],
            MatMulUnary::new(konst, t_konst, t_var, self.c_trans ^ flip, self.q_params.clone()),
        )?;
        return Ok(Some(patch));
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        cost(
            &inputs[0].shape.to_tvec(),
            &inputs[1].shape.to_tvec(),
            inputs[0].datum_type,
            self.a_trans,
            self.b_trans,
        )
    }

    typed_op_as_op!();
}

#[derive(Debug, Clone, new)]
pub struct MatMulUnary {
    a: Arc<Tensor>,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    q_params: Option<QParams>,
}

impl Op for MatMulUnary {
    fn name(&self) -> Cow<str> {
        "MatMulUnary".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut v = vec![
            format!(
                "a_trans:{:?} b_trans:{:?} c_trans:{:?}",
                self.a_trans, self.b_trans, self.c_trans
            ),
            format!("A: {:?}", self.a),
        ];
        if let Some(qp) = &self.q_params {
            v.push(format!("{:?}", qp));
        }
        Ok(v)
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for MatMulUnary {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let t = eval(
            &self.a,
            &inputs[0],
            self.a_trans,
            self.b_trans,
            self.c_trans,
            self.q_params.as_ref(),
        )?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl TypedOp for MatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            self.q_params.as_ref().map(|qp| qp.c_datum_type).unwrap_or(inputs[0].datum_type),
            &*infer_shapes(
                self.a.shape().into_iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
                inputs[0].shape.to_tvec(),
                self.a_trans,
                self.b_trans,
                self.c_trans,
            )?
            .3
        )?))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact.shape.rank() != node.outputs[0].fact.shape.rank() {
            return Ok(Invariants::none());
        }
        let mut broadcasted_a_shape: TVec<_> = self.a.shape().into();
        while broadcasted_a_shape.len() < input_fact.shape.rank() {
            broadcasted_a_shape.insert(0, 1);
        }
        let mut invars = broadcasted_a_shape[..broadcasted_a_shape.len() - 2]
            .into_iter()
            .enumerate()
            .map(|(axis, &period)| AxisInfo::simple(axis).with_period(period))
            .collect::<Vec<_>>();
        if self.b_trans && self.c_trans && input_fact.rank() >= 2 {
            invars.push(AxisInfo::simple(input_fact.shape.rank() - 2))
        }
        if !self.b_trans && !self.c_trans {
            invars.push(AxisInfo::simple(input_fact.shape.rank() - 1))
        };
        Ok(invars.into_iter().collect())
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        match change {
            AxisOp::Rm(axis) => {
                let b = &model.outlet_fact(node.inputs[0])?;
                if b.rank() > *axis + 2 && self.a.rank() <= b.rank() {
                    let op = if b.rank() - axis < self.a.rank() {
                        let mut a = self.a.clone().into_tensor();
                        a.remove_axis(*axis)?;
                        Some(Box::new(MatMulUnary { a: a.into_arc_tensor(), ..self.clone() }) as _)
                    } else {
                        None
                    };
                    Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops::array::concat::NormConcatSlice;
        use crate::ops::array::NormConcat;
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if let Some(concat) = model.node_op(node.inputs[0].node).downcast_ref::<NormConcat>() {
            let mut patch = TypedModelPatch::default();
            let k_axis = self.a.rank() - 1 - self.a_trans as usize;
            if concat.axis == input_fact.shape.rank() - 1 && self.b_trans {
                let mut input = 0;
                let concat_node = model.node(node.inputs[0].node);
                let offsets = concat
                    .offsets(&model.node_input_facts(concat_node.id)?)?
                    .iter()
                    .map(|x| x.to_integer().map(|i| i as usize))
                    .collect::<TractResult<Vec<usize>>>()?;
                let mut wires = vec![];
                for (ix, slice) in concat.slices.iter().enumerate() {
                    let wire = match slice {
                        NormConcatSlice::Const(t) => patch.add_const(
                            format!("{}-const-{}", node.name, ix),
                            t.clone().into_arc_tensor(),
                        )?,
                        NormConcatSlice::Var => {
                            input += 1;
                            patch.tap_model(model, concat_node.inputs[input - 1])?
                        }
                    };
                    let mut a = self.a.slice(k_axis, offsets[ix], offsets[ix + 1])?;
                    while a.rank() > 0 && a.shape()[0] == 1 {
                        a.remove_axis(0)?;
                    }
                    let wire = patch.wire_node(
                        format!("{}-k-{}-{}", node.name, offsets[ix], offsets[ix + 1]),
                        MatMulUnary { a: a.into_arc_tensor(), ..self.clone() },
                        &[wire],
                    )?[0];
                    wires.push(wire)
                }
                let mut wire = wires[0];
                for (ix, w) in wires[1..].iter().enumerate() {
                    wire = patch.wire_node(
                        format!("{}-k-add-{}", node.name, ix),
                        crate::ops::math::add::bin(),
                        &[wire, *w],
                    )?[0];
                }
                patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let b_fact = model.outlet_fact(node.inputs[0])?;
        let c_fact = &self.output_facts(&[b_fact])?[0];
        if axis + self.c_trans as usize == c_fact.shape.rank() {
            let a_split_axis = self.a.rank() - 1 - !self.a_trans as usize;
            let a = self.a.slice(a_split_axis, start, end)?.into_arc_tensor();
            let wire = patch.tap_model(model, node.inputs[0])?;
            return Ok(Some(
                patch.wire_node(
                    format!("{}-sliced-m-{}-{}", node.name, start, end),
                    Self { a, ..self.clone() },
                    &[wire],
                )?[0],
            ));
        }
        return Ok(None);
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        cost(
            self.a.shape(),
            &inputs[0].shape.to_tvec(),
            self.a.datum_type(),
            self.a_trans,
            self.b_trans,
        )
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
        let fact = target.outlet_fact(input)?;
        if fact.axis >= fact.shape.len() - self.a_trans as usize {
            bail!("Can not pulsify MatMulUnaryA on the k dimension");
        }
        target.wire_node(&*node.name, self.clone(), &[input])
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let b = args_1!(model.node_input_facts(node.id)?);
        if let Some(b_shape) = b.shape.as_finite() {
            let patch =
                if (self.a.datum_type(), b.datum_type) == (f32::datum_type(), f32::datum_type()) {
                    new_mat_mul_unary_finite(
                        model,
                        node,
                        self.a.clone(),
                        b_shape,
                        self.a_trans,
                        self.b_trans,
                        self.c_trans,
                        self.q_params.as_ref(),
                        &|m, k, n| MMMWrapper::Plain((tract_linalg::ops().smmm)(m, k, n)),
                    )?
                } else if (
                    self.a.datum_type(),
                    b.datum_type,
                    self.q_params.as_ref().map(|q| q.c_datum_type),
                ) == (i8::datum_type(), i8::datum_type(), Some(i8::datum_type()))
                {
                    new_mat_mul_unary_finite(
                        model,
                        node,
                        self.a.clone(),
                        b_shape,
                        self.a_trans,
                        self.b_trans,
                        self.c_trans,
                        self.q_params.as_ref(),
                        &|m, k, n| MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i8)(m, k, n)),
                    )?
                } else if (
                    self.a.datum_type(),
                    b.datum_type,
                    self.q_params.as_ref().map(|q| q.c_datum_type),
                ) == (i8::datum_type(), i8::datum_type(), Some(i32::datum_type()))
                {
                    new_mat_mul_unary_finite(
                        model,
                        node,
                        self.a.clone(),
                        b_shape,
                        self.a_trans,
                        self.b_trans,
                        self.c_trans,
                        self.q_params.as_ref(),
                        &|m, k, n| MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i32)(m, k, n)),
                    )?
                } else {
                    bail!(
                        "Unsupported combination for MatMul codegen (a: {:?}, b:{:?}, q: {:?})",
                        self.a.datum_type(),
                        b.datum_type,
                        self.q_params
                    );
                };
            return Ok(Some(patch));
        }
        Ok(None)
    }

    typed_op_as_op!();
}

impl PulsedOp for MatMulUnary {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape = infer_shapes(
            self.a.shape().into_iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            inputs[0].shape.iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?
        .2
        .iter()
        .map(|d| d.to_integer().unwrap() as usize)
        .collect::<TVec<_>>();
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}

fn new_mat_mul_unary_finite<TA, TB, TC, TI>(
    model: &TypedModel,
    node: &TypedNode,
    a: Arc<Tensor>,
    b_shape: &[usize],
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    q_params: Option<&QParams>,
    mmm: &impl Fn(usize, usize, usize) -> MMMWrapper<TA, TB, TC, TI>,
) -> TractResult<TypedModelPatch>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.tap_model(model, node.inputs[0])?;
    let mut geo = Geo::<TA, TB, TC, TI>::new(a.shape(), b_shape, a_trans, b_trans, c_trans, mmm)?;
    let a = a.to_array_view::<TA>()?;
    let a = a.into_shape(&*geo.bc_a_shape)?;
    let packed_as = Array::from_shape_fn(&a.shape()[0..a.ndim() - 2], |a_prefix| {
        let mut a = a.view();
        for x in a_prefix.slice() {
            a.index_axis_inplace(Axis(0), *x);
        }
        let mut pa = unsafe {
            Tensor::uninitialized_aligned::<TA>(
                &[geo.mm.as_mmm().a_pack().len()],
                geo.mm.as_mmm().a_pack().alignment(),
            )
            .unwrap()
        };
        geo.mm.as_mmm().a_pack().pack(
            pa.as_ptr_mut().unwrap(),
            a.as_ptr(),
            a.strides()[a_trans as usize],
            a.strides()[!a_trans as usize],
        );
        pa.into_arc_tensor()
    });
    unsafe {
        if geo.n == 1 {
            geo.mm.as_mmm_mut().b_vec_from_data_and_stride(if b_trans {
                1
            } else {
                *geo.b_shape.last().unwrap() as isize
            });
            geo.mm.as_mmm_mut().c_vec_from_data_and_stride(if c_trans {
                1
            } else {
                *geo.bc_c_shape.last().unwrap() as isize
            });
        } else {
            geo.mm.as_mmm_mut().c_from_data_and_strides(
                if c_trans { 1 } else { *geo.bc_c_shape.last().unwrap() as isize },
                if !c_trans { 1 } else { *geo.bc_c_shape.last().unwrap() as isize },
            );
        };
        if let Some(q) = q_params {
            geo.mm.set_quant_params(q)?;
        }
    }
    if geo.n > 1 {
        let mut packed_b_shape: TVec<usize> = b_shape[..b_shape.len() - 2].into();
        packed_b_shape.push(geo.mm.as_mmm().b_pack().len());
        wire = patch.wire_node(
            format!("{}-pack", &*node.name),
            phy::MatMatMulPackB {
                pack_b: geo.mm.as_mmm().b_pack().clone(),
                col_stride: if b_trans { *b_shape.last().unwrap() as isize } else { 1 },
                row_stride: if b_trans { 1 } else { *b_shape.last().unwrap() as isize },
                output_shape: packed_b_shape,
            },
            &[wire],
        )?[0];
    }
    let c_prefix_dim_and_stride = if geo.c_shape_prefix.iter().any(|d| *d > 1) {
        let c_prefix_strides: TVec<isize> = geo
            .bc_c_shape
            .iter()
            .rev()
            .scan(1isize, |s, &d| {
                let now: isize = *s;
                *s *= d as isize;
                Some(now)
            })
            .collect::<TVec<_>>()
            .into_iter()
            .skip(2)
            .rev()
            .collect::<TVec<_>>();
        Some((geo.c_shape_prefix.clone(), c_prefix_strides))
    } else {
        None
    };
    wire = patch.wire_node(
        format!("{}-matmatmul", &*node.name),
        phy::MatMatMulUnaryFinite {
            c_trans,
            bc_c_shape: geo.bc_c_shape,
            c_fact: TypedFact::dt_shape(TC::datum_type(), &*geo.final_c_shape)?,
            c_prefix_dim_and_stride,
            packed_as,
            fused_ops: None,
            mmm: geo.mm,
        },
        &[wire],
    )?[0];
    patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
    Ok(patch)
}

fn cost<A: ToDim + Clone, B: ToDim + Clone>(
    a: &[A],
    b: &[B],
    dt: DatumType,
    a_trans: bool,
    b_trans: bool,
) -> TractResult<TVec<(Cost, TDim)>> {
    let (bc_a_shape, bc_b_shape, bc_c_shape, _c_shape) = infer_shapes(
        a.iter().map(|d| d.clone().to_dim()).collect(),
        b.iter().map(|d| d.clone().to_dim()).collect(),
        a_trans,
        b_trans,
        false,
    )?;
    let mul = bc_c_shape.iter().rev().skip(2).cloned().product::<TDim>();
    let m = &bc_a_shape[bc_a_shape.len() - 2 + a_trans as usize];
    let k = &bc_a_shape[bc_a_shape.len() - 1 - a_trans as usize];
    let n = &bc_b_shape[bc_b_shape.len() - 1 - b_trans as usize];
    Ok(tvec!((Cost::FMA(dt), (mul * m * k * n))))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bin() {
        let a = rctensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = rctensor2(&[[0f32], [1.0], [2.0]]);
        let c = rctensor2(&[[5f32], [14.0]]);
        let op = MatMul::default();
        let c_found = op.eval(tvec!(a, b)).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn bin_transpose() {
        let a = rctensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = rctensor2(&[[0f32], [1.0], [2.0]]);
        let c = rctensor2(&[[5f32], [14.0]]);
        let op = MatMul::default().with_a_trans(true).with_b_trans(true).with_c_trans(true);
        let c_found = op.eval(tvec!(b, a)).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }
}
