       �K"	  �Q1(�Abrain.Event:24íq��     G��	ӭ�Q1(�A"��
t
	inputs/xsPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
t
	inputs/ysPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
h
hidden_input/2_2D/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
hidden_input/2_2DReshape	inputs/xshidden_input/2_2D/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
4hidden_input/Weights/Initializer/random_normal/shapeConst*'
_class
loc:@hidden_input/Weights*
valueB"   
   *
dtype0*
_output_shapes
:
�
3hidden_input/Weights/Initializer/random_normal/meanConst*'
_class
loc:@hidden_input/Weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
5hidden_input/Weights/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *'
_class
loc:@hidden_input/Weights*
valueB
 *  �?
�
Chidden_input/Weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal4hidden_input/Weights/Initializer/random_normal/shape*
T0*'
_class
loc:@hidden_input/Weights*
seed2 *
dtype0*
_output_shapes

:
*

seed 
�
2hidden_input/Weights/Initializer/random_normal/mulMulChidden_input/Weights/Initializer/random_normal/RandomStandardNormal5hidden_input/Weights/Initializer/random_normal/stddev*
_output_shapes

:
*
T0*'
_class
loc:@hidden_input/Weights
�
.hidden_input/Weights/Initializer/random_normalAdd2hidden_input/Weights/Initializer/random_normal/mul3hidden_input/Weights/Initializer/random_normal/mean*
T0*'
_class
loc:@hidden_input/Weights*
_output_shapes

:

�
hidden_input/Weights
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *'
_class
loc:@hidden_input/Weights*
	container *
shape
:

�
hidden_input/Weights/AssignAssignhidden_input/Weights.hidden_input/Weights/Initializer/random_normal*
use_locking(*
T0*'
_class
loc:@hidden_input/Weights*
validate_shape(*
_output_shapes

:

�
hidden_input/Weights/readIdentityhidden_input/Weights*
_output_shapes

:
*
T0*'
_class
loc:@hidden_input/Weights
�
%hidden_input/biases/Initializer/ConstConst*&
_class
loc:@hidden_input/biases*
valueB
*���=*
dtype0*
_output_shapes
:

�
hidden_input/biases
VariableV2*
shared_name *&
_class
loc:@hidden_input/biases*
	container *
shape:
*
dtype0*
_output_shapes
:

�
hidden_input/biases/AssignAssignhidden_input/biases%hidden_input/biases/Initializer/Const*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*&
_class
loc:@hidden_input/biases
�
hidden_input/biases/readIdentityhidden_input/biases*
T0*&
_class
loc:@hidden_input/biases*
_output_shapes
:

�
hidden_input/ys_in/MatMulMatMulhidden_input/2_2Dhidden_input/Weights/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
hidden_input/ys_in/AddAddhidden_input/ys_in/MatMulhidden_input/biases/read*
T0*'
_output_shapes
:���������

l
hidden_input/2_3D/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
�
hidden_input/2_3DReshapehidden_input/ys_in/Addhidden_input/2_3D/shape*
T0*
Tshape0*+
_output_shapes
:���������

y
/cell/initial_state/BasicLSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:2
{
1cell/initial_state/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
w
5cell/initial_state/BasicLSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
0cell/initial_state/BasicLSTMCellZeroState/concatConcatV2/cell/initial_state/BasicLSTMCellZeroState/Const1cell/initial_state/BasicLSTMCellZeroState/Const_15cell/initial_state/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
z
5cell/initial_state/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/cell/initial_state/BasicLSTMCellZeroState/zerosFill0cell/initial_state/BasicLSTMCellZeroState/concat5cell/initial_state/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes

:2

{
1cell/initial_state/BasicLSTMCellZeroState/Const_2Const*
valueB:2*
dtype0*
_output_shapes
:
{
1cell/initial_state/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:

{
1cell/initial_state/BasicLSTMCellZeroState/Const_4Const*
dtype0*
_output_shapes
:*
valueB:2
{
1cell/initial_state/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
y
7cell/initial_state/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
2cell/initial_state/BasicLSTMCellZeroState/concat_1ConcatV21cell/initial_state/BasicLSTMCellZeroState/Const_41cell/initial_state/BasicLSTMCellZeroState/Const_57cell/initial_state/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
|
7cell/initial_state/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1cell/initial_state/BasicLSTMCellZeroState/zeros_1Fill2cell/initial_state/BasicLSTMCellZeroState/concat_17cell/initial_state/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes

:2

{
1cell/initial_state/BasicLSTMCellZeroState/Const_6Const*
valueB:2*
dtype0*
_output_shapes
:
{
1cell/initial_state/BasicLSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:

O
cell/rnn/RankConst*
dtype0*
_output_shapes
: *
value	B :
V
cell/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
V
cell/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
z
cell/rnn/rangeRangecell/rnn/range/startcell/rnn/Rankcell/rnn/range/delta*

Tidx0*
_output_shapes
:
i
cell/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
V
cell/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/concatConcatV2cell/rnn/concat/values_0cell/rnn/rangecell/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
cell/rnn/transpose	Transposehidden_input/2_3Dcell/rnn/concat*
Tperm0*
T0*+
_output_shapes
:���������

`
cell/rnn/ShapeShapecell/rnn/transpose*
_output_shapes
:*
T0*
out_type0
f
cell/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
h
cell/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
cell/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
cell/rnn/strided_sliceStridedSlicecell/rnn/Shapecell/rnn/strided_slice/stackcell/rnn/strided_slice/stack_1cell/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
b
cell/rnn/Shape_1Shapecell/rnn/transpose*
T0*
out_type0*
_output_shapes
:
h
cell/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
j
 cell/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
j
 cell/rnn/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
cell/rnn/strided_slice_1StridedSlicecell/rnn/Shape_1cell/rnn/strided_slice_1/stack cell/rnn/strided_slice_1/stack_1 cell/rnn/strided_slice_1/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
b
cell/rnn/Shape_2Shapecell/rnn/transpose*
_output_shapes
:*
T0*
out_type0
h
cell/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
j
 cell/rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
j
 cell/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
cell/rnn/strided_slice_2StridedSlicecell/rnn/Shape_2cell/rnn/strided_slice_2/stack cell/rnn/strided_slice_2/stack_1 cell/rnn/strided_slice_2/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
Y
cell/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/ExpandDims
ExpandDimscell/rnn/strided_slice_2cell/rnn/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
X
cell/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB:

X
cell/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/concat_1ConcatV2cell/rnn/ExpandDimscell/rnn/Constcell/rnn/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
Y
cell/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
cell/rnn/zerosFillcell/rnn/concat_1cell/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������

O
cell/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
�
cell/rnn/TensorArrayTensorArrayV3cell/rnn/strided_slice_1*4
tensor_array_namecell/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
�
cell/rnn/TensorArray_1TensorArrayV3cell/rnn/strided_slice_1*3
tensor_array_namecell/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
s
!cell/rnn/TensorArrayUnstack/ShapeShapecell/rnn/transpose*
T0*
out_type0*
_output_shapes
:
y
/cell/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
{
1cell/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
{
1cell/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
)cell/rnn/TensorArrayUnstack/strided_sliceStridedSlice!cell/rnn/TensorArrayUnstack/Shape/cell/rnn/TensorArrayUnstack/strided_slice/stack1cell/rnn/TensorArrayUnstack/strided_slice/stack_11cell/rnn/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
i
'cell/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
i
'cell/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
!cell/rnn/TensorArrayUnstack/rangeRange'cell/rnn/TensorArrayUnstack/range/start)cell/rnn/TensorArrayUnstack/strided_slice'cell/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
Ccell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3cell/rnn/TensorArray_1!cell/rnn/TensorArrayUnstack/rangecell/rnn/transposecell/rnn/TensorArray_1:1*
T0*%
_class
loc:@cell/rnn/transpose*
_output_shapes
: 
T
cell/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
j
cell/rnn/MaximumMaximumcell/rnn/Maximum/xcell/rnn/strided_slice_1*
T0*
_output_shapes
: 
h
cell/rnn/MinimumMinimumcell/rnn/strided_slice_1cell/rnn/Maximum*
T0*
_output_shapes
: 
b
 cell/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/while/EnterEnter cell/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
cell/rnn/while/Enter_1Entercell/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
cell/rnn/while/Enter_2Entercell/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
cell/rnn/while/Enter_3Enter/cell/initial_state/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:2
*,

frame_namecell/rnn/while/while_context
�
cell/rnn/while/Enter_4Enter1cell/initial_state/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:2
*,

frame_namecell/rnn/while/while_context
}
cell/rnn/while/MergeMergecell/rnn/while/Entercell/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
cell/rnn/while/Merge_1Mergecell/rnn/while/Enter_1cell/rnn/while/NextIteration_1*
N*
_output_shapes
: : *
T0
�
cell/rnn/while/Merge_2Mergecell/rnn/while/Enter_2cell/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
cell/rnn/while/Merge_3Mergecell/rnn/while/Enter_3cell/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:2
: 
�
cell/rnn/while/Merge_4Mergecell/rnn/while/Enter_4cell/rnn/while/NextIteration_4*
T0*
N* 
_output_shapes
:2
: 
m
cell/rnn/while/LessLesscell/rnn/while/Mergecell/rnn/while/Less/Enter*
_output_shapes
: *
T0
�
cell/rnn/while/Less/EnterEntercell/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
s
cell/rnn/while/Less_1Lesscell/rnn/while/Merge_1cell/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
cell/rnn/while/Less_1/EnterEntercell/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
k
cell/rnn/while/LogicalAnd
LogicalAndcell/rnn/while/Lesscell/rnn/while/Less_1*
_output_shapes
: 
V
cell/rnn/while/LoopCondLoopCondcell/rnn/while/LogicalAnd*
_output_shapes
: 
�
cell/rnn/while/SwitchSwitchcell/rnn/while/Mergecell/rnn/while/LoopCond*
T0*'
_class
loc:@cell/rnn/while/Merge*
_output_shapes
: : 
�
cell/rnn/while/Switch_1Switchcell/rnn/while/Merge_1cell/rnn/while/LoopCond*
T0*)
_class
loc:@cell/rnn/while/Merge_1*
_output_shapes
: : 
�
cell/rnn/while/Switch_2Switchcell/rnn/while/Merge_2cell/rnn/while/LoopCond*
T0*)
_class
loc:@cell/rnn/while/Merge_2*
_output_shapes
: : 
�
cell/rnn/while/Switch_3Switchcell/rnn/while/Merge_3cell/rnn/while/LoopCond*(
_output_shapes
:2
:2
*
T0*)
_class
loc:@cell/rnn/while/Merge_3
�
cell/rnn/while/Switch_4Switchcell/rnn/while/Merge_4cell/rnn/while/LoopCond*
T0*)
_class
loc:@cell/rnn/while/Merge_4*(
_output_shapes
:2
:2

]
cell/rnn/while/IdentityIdentitycell/rnn/while/Switch:1*
_output_shapes
: *
T0
a
cell/rnn/while/Identity_1Identitycell/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
a
cell/rnn/while/Identity_2Identitycell/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
i
cell/rnn/while/Identity_3Identitycell/rnn/while/Switch_3:1*
_output_shapes

:2
*
T0
i
cell/rnn/while/Identity_4Identitycell/rnn/while/Switch_4:1*
T0*
_output_shapes

:2

p
cell/rnn/while/add/yConst^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
cell/rnn/while/addAddcell/rnn/while/Identitycell/rnn/while/add/y*
T0*
_output_shapes
: 
�
 cell/rnn/while/TensorArrayReadV3TensorArrayReadV3&cell/rnn/while/TensorArrayReadV3/Entercell/rnn/while/Identity_1(cell/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
&cell/rnn/while/TensorArrayReadV3/EnterEntercell/rnn/TensorArray_1*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context*
T0*
is_constant(
�
(cell/rnn/while/TensorArrayReadV3/Enter_1EnterCcell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
@cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB
 *�衾
�
>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
Hcell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform@cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:(*

seed *
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
seed2 
�
>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSub>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/max>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
_output_shapes
: 
�
>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulHcell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
_output_shapes

:(*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
:cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAdd>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mul>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes

:(*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
cell/rnn/basic_lstm_cell/kernel
VariableV2*
	container *
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
&cell/rnn/basic_lstm_cell/kernel/AssignAssigncell/rnn/basic_lstm_cell/kernel:cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
z
$cell/rnn/basic_lstm_cell/kernel/readIdentitycell/rnn/basic_lstm_cell/kernel*
_output_shapes

:(*
T0
�
/cell/rnn/basic_lstm_cell/bias/Initializer/zerosConst*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
cell/rnn/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
	container *
shape:(
�
$cell/rnn/basic_lstm_cell/bias/AssignAssigncell/rnn/basic_lstm_cell/bias/cell/rnn/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
r
"cell/rnn/basic_lstm_cell/bias/readIdentitycell/rnn/basic_lstm_cell/bias*
_output_shapes
:(*
T0
�
$cell/rnn/while/basic_lstm_cell/ConstConst^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
*cell/rnn/while/basic_lstm_cell/concat/axisConst^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
%cell/rnn/while/basic_lstm_cell/concatConcatV2 cell/rnn/while/TensorArrayReadV3cell/rnn/while/Identity_4*cell/rnn/while/basic_lstm_cell/concat/axis*
N*
_output_shapes

:2*

Tidx0*
T0
�
%cell/rnn/while/basic_lstm_cell/MatMulMatMul%cell/rnn/while/basic_lstm_cell/concat+cell/rnn/while/basic_lstm_cell/MatMul/Enter*
_output_shapes

:2(*
transpose_a( *
transpose_b( *
T0
�
+cell/rnn/while/basic_lstm_cell/MatMul/EnterEnter$cell/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*,

frame_namecell/rnn/while/while_context
�
&cell/rnn/while/basic_lstm_cell/BiasAddBiasAdd%cell/rnn/while/basic_lstm_cell/MatMul,cell/rnn/while/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:2(
�
,cell/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter"cell/rnn/basic_lstm_cell/bias/read*
parallel_iterations *
_output_shapes
:(*,

frame_namecell/rnn/while/while_context*
T0*
is_constant(
�
&cell/rnn/while/basic_lstm_cell/Const_1Const^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
$cell/rnn/while/basic_lstm_cell/splitSplit$cell/rnn/while/basic_lstm_cell/Const&cell/rnn/while/basic_lstm_cell/BiasAdd*<
_output_shapes*
(:2
:2
:2
:2
*
	num_split*
T0
�
&cell/rnn/while/basic_lstm_cell/Const_2Const^cell/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"cell/rnn/while/basic_lstm_cell/AddAdd&cell/rnn/while/basic_lstm_cell/split:2&cell/rnn/while/basic_lstm_cell/Const_2*
T0*
_output_shapes

:2

~
&cell/rnn/while/basic_lstm_cell/SigmoidSigmoid"cell/rnn/while/basic_lstm_cell/Add*
T0*
_output_shapes

:2

�
"cell/rnn/while/basic_lstm_cell/MulMulcell/rnn/while/Identity_3&cell/rnn/while/basic_lstm_cell/Sigmoid*
_output_shapes

:2
*
T0
�
(cell/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid$cell/rnn/while/basic_lstm_cell/split*
T0*
_output_shapes

:2

|
#cell/rnn/while/basic_lstm_cell/TanhTanh&cell/rnn/while/basic_lstm_cell/split:1*
_output_shapes

:2
*
T0
�
$cell/rnn/while/basic_lstm_cell/Mul_1Mul(cell/rnn/while/basic_lstm_cell/Sigmoid_1#cell/rnn/while/basic_lstm_cell/Tanh*
T0*
_output_shapes

:2

�
$cell/rnn/while/basic_lstm_cell/Add_1Add"cell/rnn/while/basic_lstm_cell/Mul$cell/rnn/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes

:2

|
%cell/rnn/while/basic_lstm_cell/Tanh_1Tanh$cell/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:2

�
(cell/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid&cell/rnn/while/basic_lstm_cell/split:3*
T0*
_output_shapes

:2

�
$cell/rnn/while/basic_lstm_cell/Mul_2Mul%cell/rnn/while/basic_lstm_cell/Tanh_1(cell/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:2

�
2cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV38cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entercell/rnn/while/Identity_1$cell/rnn/while/basic_lstm_cell/Mul_2cell/rnn/while/Identity_2*
_output_shapes
: *
T0*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2
�
8cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntercell/rnn/TensorArray*
is_constant(*
_output_shapes
:*,

frame_namecell/rnn/while/while_context*
T0*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations 
r
cell/rnn/while/add_1/yConst^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
cell/rnn/while/add_1Addcell/rnn/while/Identity_1cell/rnn/while/add_1/y*
T0*
_output_shapes
: 
b
cell/rnn/while/NextIterationNextIterationcell/rnn/while/add*
T0*
_output_shapes
: 
f
cell/rnn/while/NextIteration_1NextIterationcell/rnn/while/add_1*
T0*
_output_shapes
: 
�
cell/rnn/while/NextIteration_2NextIteration2cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
~
cell/rnn/while/NextIteration_3NextIteration$cell/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:2

~
cell/rnn/while/NextIteration_4NextIteration$cell/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes

:2
*
T0
S
cell/rnn/while/ExitExitcell/rnn/while/Switch*
T0*
_output_shapes
: 
W
cell/rnn/while/Exit_1Exitcell/rnn/while/Switch_1*
_output_shapes
: *
T0
W
cell/rnn/while/Exit_2Exitcell/rnn/while/Switch_2*
_output_shapes
: *
T0
_
cell/rnn/while/Exit_3Exitcell/rnn/while/Switch_3*
T0*
_output_shapes

:2

_
cell/rnn/while/Exit_4Exitcell/rnn/while/Switch_4*
T0*
_output_shapes

:2

�
+cell/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3cell/rnn/TensorArraycell/rnn/while/Exit_2*
_output_shapes
: *'
_class
loc:@cell/rnn/TensorArray
�
%cell/rnn/TensorArrayStack/range/startConst*'
_class
loc:@cell/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
%cell/rnn/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *'
_class
loc:@cell/rnn/TensorArray*
value	B :
�
cell/rnn/TensorArrayStack/rangeRange%cell/rnn/TensorArrayStack/range/start+cell/rnn/TensorArrayStack/TensorArraySizeV3%cell/rnn/TensorArrayStack/range/delta*#
_output_shapes
:���������*

Tidx0*'
_class
loc:@cell/rnn/TensorArray
�
-cell/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3cell/rnn/TensorArraycell/rnn/TensorArrayStack/rangecell/rnn/while/Exit_2*'
_class
loc:@cell/rnn/TensorArray*
dtype0*"
_output_shapes
:2
*
element_shape
:2

Z
cell/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB:

Q
cell/rnn/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
X
cell/rnn/range_1/startConst*
dtype0*
_output_shapes
: *
value	B :
X
cell/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
cell/rnn/range_1Rangecell/rnn/range_1/startcell/rnn/Rank_1cell/rnn/range_1/delta*
_output_shapes
:*

Tidx0
k
cell/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
X
cell/rnn/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
cell/rnn/concat_2ConcatV2cell/rnn/concat_2/values_0cell/rnn/range_1cell/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
cell/rnn/transpose_1	Transpose-cell/rnn/TensorArrayStack/TensorArrayGatherV3cell/rnn/concat_2*
T0*"
_output_shapes
:2
*
Tperm0
i
hidden_output/2_2D/shapeConst*
valueB"����
   *
dtype0*
_output_shapes
:
�
hidden_output/2_2DReshapecell/rnn/transpose_1hidden_output/2_2D/shape*
_output_shapes
:	�
*
T0*
Tshape0
�
5hidden_output/Weights/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*(
_class
loc:@hidden_output/Weights*
valueB"
      
�
4hidden_output/Weights/Initializer/random_normal/meanConst*(
_class
loc:@hidden_output/Weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6hidden_output/Weights/Initializer/random_normal/stddevConst*(
_class
loc:@hidden_output/Weights*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dhidden_output/Weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal5hidden_output/Weights/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:
*

seed *
T0*(
_class
loc:@hidden_output/Weights
�
3hidden_output/Weights/Initializer/random_normal/mulMulDhidden_output/Weights/Initializer/random_normal/RandomStandardNormal6hidden_output/Weights/Initializer/random_normal/stddev*
_output_shapes

:
*
T0*(
_class
loc:@hidden_output/Weights
�
/hidden_output/Weights/Initializer/random_normalAdd3hidden_output/Weights/Initializer/random_normal/mul4hidden_output/Weights/Initializer/random_normal/mean*
T0*(
_class
loc:@hidden_output/Weights*
_output_shapes

:

�
hidden_output/Weights
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *(
_class
loc:@hidden_output/Weights*
	container *
shape
:

�
hidden_output/Weights/AssignAssignhidden_output/Weights/hidden_output/Weights/Initializer/random_normal*
T0*(
_class
loc:@hidden_output/Weights*
validate_shape(*
_output_shapes

:
*
use_locking(
�
hidden_output/Weights/readIdentityhidden_output/Weights*
_output_shapes

:
*
T0*(
_class
loc:@hidden_output/Weights
�
&hidden_output/biases/Initializer/ConstConst*'
_class
loc:@hidden_output/biases*
valueB*���=*
dtype0*
_output_shapes
:
�
hidden_output/biases
VariableV2*'
_class
loc:@hidden_output/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
hidden_output/biases/AssignAssignhidden_output/biases&hidden_output/biases/Initializer/Const*
T0*'
_class
loc:@hidden_output/biases*
validate_shape(*
_output_shapes
:*
use_locking(
�
hidden_output/biases/readIdentityhidden_output/biases*
T0*'
_class
loc:@hidden_output/biases*
_output_shapes
:
�
hidden_output/ys_out/MatMulMatMulhidden_output/2_2Dhidden_output/Weights/read*
T0*
_output_shapes
:	�*
transpose_a( *
transpose_b( 
�
hidden_output/ys_out/AddAddhidden_output/ys_out/MatMulhidden_output/biases/read*
T0*
_output_shapes
:	�
p
loss/reshape_prediction/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
loss/reshape_predictionReshapehidden_output/ys_out/Addloss/reshape_prediction/shape*
T0*
Tshape0*
_output_shapes	
:�
l
loss/reshape_target/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
loss/reshape_targetReshape	inputs/ysloss/reshape_target/shape*
T0*
Tshape0*#
_output_shapes
:���������
d
loss/ones/shape_as_tensorConst*
valueB:�*
dtype0*
_output_shapes
:
T
loss/ones/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
u
	loss/onesFillloss/ones/shape_as_tensorloss/ones/Const*
T0*

index_type0*
_output_shapes	
:�
j
loss/losses/SubSubloss/reshape_targetloss/reshape_prediction*
T0*
_output_shapes	
:�
S
loss/losses/SquareSquareloss/losses/Sub*
_output_shapes	
:�*
T0
[
loss/losses/mulMulloss/losses/Square	loss/ones*
T0*
_output_shapes	
:�
V
loss/losses/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
Z
loss/losses/addAdd	loss/onesloss/losses/add/y*
T0*
_output_shapes	
:�
f
loss/losses/truedivRealDivloss/losses/mulloss/losses/add*
_output_shapes	
:�*
T0
a
loss/average_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/average_loss/sum_lossSumloss/losses/truedivloss/average_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
 loss/average_loss/average_loss/yConst*
valueB
 *  HB*
dtype0*
_output_shapes
: 
�
loss/average_loss/average_lossRealDivloss/average_loss/sum_loss loss/average_loss/average_loss/y*
T0*
_output_shapes
: 
r
loss/average_loss/loss/tagsConst*'
valueB Bloss/average_loss/loss*
dtype0*
_output_shapes
: 
�
loss/average_loss/lossScalarSummaryloss/average_loss/loss/tagsloss/average_loss/average_loss*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Y
train/gradients/f_countConst*
dtype0*
_output_shapes
: *
value	B : 
�
train/gradients/f_count_1Entertrain/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
s
train/gradients/SwitchSwitchtrain/gradients/Mergecell/rnn/while/LoopCond*
T0*
_output_shapes
: : 
q
train/gradients/Add/yConst^cell/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
l
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
T0*
_output_shapes
: 
�
train/gradients/NextIterationNextIterationtrain/gradients/Addf^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2P^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2J^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2L^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2J^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2L^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2H^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2J^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2N^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2P^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
_output_shapes
: *
T0
Z
train/gradients/f_count_2Exittrain/gradients/Switch*
T0*
_output_shapes
: 
Y
train/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context
�
train/gradients/Merge_1Mergetrain/gradients/b_count_1train/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
train/gradients/GreaterEqualGreaterEqualtrain/gradients/Merge_1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
"train/gradients/GreaterEqual/EnterEntertrain/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context
[
train/gradients/b_count_2LoopCondtrain/gradients/GreaterEqual*
_output_shapes
: 
y
train/gradients/Switch_1Switchtrain/gradients/Merge_1train/gradients/b_count_2*
_output_shapes
: : *
T0
{
train/gradients/SubSubtrain/gradients/Switch_1:1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
train/gradients/NextIteration_1NextIterationtrain/gradients/Suba^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
\
train/gradients/b_count_3Exittrain/gradients/Switch_1*
_output_shapes
: *
T0
|
9train/gradients/loss/average_loss/average_loss_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
~
;train/gradients/loss/average_loss/average_loss_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Itrain/gradients/loss/average_loss/average_loss_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/loss/average_loss/average_loss_grad/Shape;train/gradients/loss/average_loss/average_loss_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
;train/gradients/loss/average_loss/average_loss_grad/RealDivRealDivtrain/gradients/Fill loss/average_loss/average_loss/y*
T0*
_output_shapes
: 
�
7train/gradients/loss/average_loss/average_loss_grad/SumSum;train/gradients/loss/average_loss/average_loss_grad/RealDivItrain/gradients/loss/average_loss/average_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;train/gradients/loss/average_loss/average_loss_grad/ReshapeReshape7train/gradients/loss/average_loss/average_loss_grad/Sum9train/gradients/loss/average_loss/average_loss_grad/Shape*
_output_shapes
: *
T0*
Tshape0
{
7train/gradients/loss/average_loss/average_loss_grad/NegNegloss/average_loss/sum_loss*
_output_shapes
: *
T0
�
=train/gradients/loss/average_loss/average_loss_grad/RealDiv_1RealDiv7train/gradients/loss/average_loss/average_loss_grad/Neg loss/average_loss/average_loss/y*
T0*
_output_shapes
: 
�
=train/gradients/loss/average_loss/average_loss_grad/RealDiv_2RealDiv=train/gradients/loss/average_loss/average_loss_grad/RealDiv_1 loss/average_loss/average_loss/y*
T0*
_output_shapes
: 
�
7train/gradients/loss/average_loss/average_loss_grad/mulMultrain/gradients/Fill=train/gradients/loss/average_loss/average_loss_grad/RealDiv_2*
T0*
_output_shapes
: 
�
9train/gradients/loss/average_loss/average_loss_grad/Sum_1Sum7train/gradients/loss/average_loss/average_loss_grad/mulKtrain/gradients/loss/average_loss/average_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
=train/gradients/loss/average_loss/average_loss_grad/Reshape_1Reshape9train/gradients/loss/average_loss/average_loss_grad/Sum_1;train/gradients/loss/average_loss/average_loss_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Dtrain/gradients/loss/average_loss/average_loss_grad/tuple/group_depsNoOp<^train/gradients/loss/average_loss/average_loss_grad/Reshape>^train/gradients/loss/average_loss/average_loss_grad/Reshape_1
�
Ltrain/gradients/loss/average_loss/average_loss_grad/tuple/control_dependencyIdentity;train/gradients/loss/average_loss/average_loss_grad/ReshapeE^train/gradients/loss/average_loss/average_loss_grad/tuple/group_deps*
_output_shapes
: *
T0*N
_classD
B@loc:@train/gradients/loss/average_loss/average_loss_grad/Reshape
�
Ntrain/gradients/loss/average_loss/average_loss_grad/tuple/control_dependency_1Identity=train/gradients/loss/average_loss/average_loss_grad/Reshape_1E^train/gradients/loss/average_loss/average_loss_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/loss/average_loss/average_loss_grad/Reshape_1*
_output_shapes
: 
�
=train/gradients/loss/average_loss/sum_loss_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
7train/gradients/loss/average_loss/sum_loss_grad/ReshapeReshapeLtrain/gradients/loss/average_loss/average_loss_grad/tuple/control_dependency=train/gradients/loss/average_loss/sum_loss_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
5train/gradients/loss/average_loss/sum_loss_grad/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
�
4train/gradients/loss/average_loss/sum_loss_grad/TileTile7train/gradients/loss/average_loss/sum_loss_grad/Reshape5train/gradients/loss/average_loss/sum_loss_grad/Const*

Tmultiples0*
T0*
_output_shapes	
:�
y
.train/gradients/loss/losses/truediv_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
{
0train/gradients/loss/losses/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
>train/gradients/loss/losses/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs.train/gradients/loss/losses/truediv_grad/Shape0train/gradients/loss/losses/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0train/gradients/loss/losses/truediv_grad/RealDivRealDiv4train/gradients/loss/average_loss/sum_loss_grad/Tileloss/losses/add*
_output_shapes	
:�*
T0
�
,train/gradients/loss/losses/truediv_grad/SumSum0train/gradients/loss/losses/truediv_grad/RealDiv>train/gradients/loss/losses/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
0train/gradients/loss/losses/truediv_grad/ReshapeReshape,train/gradients/loss/losses/truediv_grad/Sum.train/gradients/loss/losses/truediv_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
j
,train/gradients/loss/losses/truediv_grad/NegNegloss/losses/mul*
T0*
_output_shapes	
:�
�
2train/gradients/loss/losses/truediv_grad/RealDiv_1RealDiv,train/gradients/loss/losses/truediv_grad/Negloss/losses/add*
T0*
_output_shapes	
:�
�
2train/gradients/loss/losses/truediv_grad/RealDiv_2RealDiv2train/gradients/loss/losses/truediv_grad/RealDiv_1loss/losses/add*
_output_shapes	
:�*
T0
�
,train/gradients/loss/losses/truediv_grad/mulMul4train/gradients/loss/average_loss/sum_loss_grad/Tile2train/gradients/loss/losses/truediv_grad/RealDiv_2*
T0*
_output_shapes	
:�
�
.train/gradients/loss/losses/truediv_grad/Sum_1Sum,train/gradients/loss/losses/truediv_grad/mul@train/gradients/loss/losses/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
2train/gradients/loss/losses/truediv_grad/Reshape_1Reshape.train/gradients/loss/losses/truediv_grad/Sum_10train/gradients/loss/losses/truediv_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
9train/gradients/loss/losses/truediv_grad/tuple/group_depsNoOp1^train/gradients/loss/losses/truediv_grad/Reshape3^train/gradients/loss/losses/truediv_grad/Reshape_1
�
Atrain/gradients/loss/losses/truediv_grad/tuple/control_dependencyIdentity0train/gradients/loss/losses/truediv_grad/Reshape:^train/gradients/loss/losses/truediv_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/loss/losses/truediv_grad/Reshape*
_output_shapes	
:�
�
Ctrain/gradients/loss/losses/truediv_grad/tuple/control_dependency_1Identity2train/gradients/loss/losses/truediv_grad/Reshape_1:^train/gradients/loss/losses/truediv_grad/tuple/group_deps*
T0*E
_class;
97loc:@train/gradients/loss/losses/truediv_grad/Reshape_1*
_output_shapes	
:�
�
(train/gradients/loss/losses/mul_grad/MulMulAtrain/gradients/loss/losses/truediv_grad/tuple/control_dependency	loss/ones*
T0*
_output_shapes	
:�
�
*train/gradients/loss/losses/mul_grad/Mul_1MulAtrain/gradients/loss/losses/truediv_grad/tuple/control_dependencyloss/losses/Square*
T0*
_output_shapes	
:�
�
5train/gradients/loss/losses/mul_grad/tuple/group_depsNoOp)^train/gradients/loss/losses/mul_grad/Mul+^train/gradients/loss/losses/mul_grad/Mul_1
�
=train/gradients/loss/losses/mul_grad/tuple/control_dependencyIdentity(train/gradients/loss/losses/mul_grad/Mul6^train/gradients/loss/losses/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*;
_class1
/-loc:@train/gradients/loss/losses/mul_grad/Mul
�
?train/gradients/loss/losses/mul_grad/tuple/control_dependency_1Identity*train/gradients/loss/losses/mul_grad/Mul_16^train/gradients/loss/losses/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/loss/losses/mul_grad/Mul_1*
_output_shapes	
:�
�
-train/gradients/loss/losses/Square_grad/ConstConst>^train/gradients/loss/losses/mul_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
+train/gradients/loss/losses/Square_grad/MulMulloss/losses/Sub-train/gradients/loss/losses/Square_grad/Const*
T0*
_output_shapes	
:�
�
-train/gradients/loss/losses/Square_grad/Mul_1Mul=train/gradients/loss/losses/mul_grad/tuple/control_dependency+train/gradients/loss/losses/Square_grad/Mul*
T0*
_output_shapes	
:�
}
*train/gradients/loss/losses/Sub_grad/ShapeShapeloss/reshape_target*
T0*
out_type0*
_output_shapes
:
w
,train/gradients/loss/losses/Sub_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
:train/gradients/loss/losses/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/loss/losses/Sub_grad/Shape,train/gradients/loss/losses/Sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
(train/gradients/loss/losses/Sub_grad/SumSum-train/gradients/loss/losses/Square_grad/Mul_1:train/gradients/loss/losses/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
,train/gradients/loss/losses/Sub_grad/ReshapeReshape(train/gradients/loss/losses/Sub_grad/Sum*train/gradients/loss/losses/Sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
*train/gradients/loss/losses/Sub_grad/Sum_1Sum-train/gradients/loss/losses/Square_grad/Mul_1<train/gradients/loss/losses/Sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
~
(train/gradients/loss/losses/Sub_grad/NegNeg*train/gradients/loss/losses/Sub_grad/Sum_1*
_output_shapes
:*
T0
�
.train/gradients/loss/losses/Sub_grad/Reshape_1Reshape(train/gradients/loss/losses/Sub_grad/Neg,train/gradients/loss/losses/Sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
5train/gradients/loss/losses/Sub_grad/tuple/group_depsNoOp-^train/gradients/loss/losses/Sub_grad/Reshape/^train/gradients/loss/losses/Sub_grad/Reshape_1
�
=train/gradients/loss/losses/Sub_grad/tuple/control_dependencyIdentity,train/gradients/loss/losses/Sub_grad/Reshape6^train/gradients/loss/losses/Sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/loss/losses/Sub_grad/Reshape*#
_output_shapes
:���������
�
?train/gradients/loss/losses/Sub_grad/tuple/control_dependency_1Identity.train/gradients/loss/losses/Sub_grad/Reshape_16^train/gradients/loss/losses/Sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*A
_class7
53loc:@train/gradients/loss/losses/Sub_grad/Reshape_1
�
2train/gradients/loss/reshape_prediction_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�     
�
4train/gradients/loss/reshape_prediction_grad/ReshapeReshape?train/gradients/loss/losses/Sub_grad/tuple/control_dependency_12train/gradients/loss/reshape_prediction_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
3train/gradients/hidden_output/ys_out/Add_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:

5train/gradients/hidden_output/ys_out/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
Ctrain/gradients/hidden_output/ys_out/Add_grad/BroadcastGradientArgsBroadcastGradientArgs3train/gradients/hidden_output/ys_out/Add_grad/Shape5train/gradients/hidden_output/ys_out/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1train/gradients/hidden_output/ys_out/Add_grad/SumSum4train/gradients/loss/reshape_prediction_grad/ReshapeCtrain/gradients/hidden_output/ys_out/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/hidden_output/ys_out/Add_grad/ReshapeReshape1train/gradients/hidden_output/ys_out/Add_grad/Sum3train/gradients/hidden_output/ys_out/Add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
3train/gradients/hidden_output/ys_out/Add_grad/Sum_1Sum4train/gradients/loss/reshape_prediction_grad/ReshapeEtrain/gradients/hidden_output/ys_out/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7train/gradients/hidden_output/ys_out/Add_grad/Reshape_1Reshape3train/gradients/hidden_output/ys_out/Add_grad/Sum_15train/gradients/hidden_output/ys_out/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
>train/gradients/hidden_output/ys_out/Add_grad/tuple/group_depsNoOp6^train/gradients/hidden_output/ys_out/Add_grad/Reshape8^train/gradients/hidden_output/ys_out/Add_grad/Reshape_1
�
Ftrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependencyIdentity5train/gradients/hidden_output/ys_out/Add_grad/Reshape?^train/gradients/hidden_output/ys_out/Add_grad/tuple/group_deps*
_output_shapes
:	�*
T0*H
_class>
<:loc:@train/gradients/hidden_output/ys_out/Add_grad/Reshape
�
Htrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependency_1Identity7train/gradients/hidden_output/ys_out/Add_grad/Reshape_1?^train/gradients/hidden_output/ys_out/Add_grad/tuple/group_deps*
T0*J
_class@
><loc:@train/gradients/hidden_output/ys_out/Add_grad/Reshape_1*
_output_shapes
:
�
7train/gradients/hidden_output/ys_out/MatMul_grad/MatMulMatMulFtrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependencyhidden_output/Weights/read*
T0*
_output_shapes
:	�
*
transpose_a( *
transpose_b(
�
9train/gradients/hidden_output/ys_out/MatMul_grad/MatMul_1MatMulhidden_output/2_2DFtrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
Atrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/group_depsNoOp8^train/gradients/hidden_output/ys_out/MatMul_grad/MatMul:^train/gradients/hidden_output/ys_out/MatMul_grad/MatMul_1
�
Itrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/control_dependencyIdentity7train/gradients/hidden_output/ys_out/MatMul_grad/MatMulB^train/gradients/hidden_output/ys_out/MatMul_grad/tuple/group_deps*
_output_shapes
:	�
*
T0*J
_class@
><loc:@train/gradients/hidden_output/ys_out/MatMul_grad/MatMul
�
Ktrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/control_dependency_1Identity9train/gradients/hidden_output/ys_out/MatMul_grad/MatMul_1B^train/gradients/hidden_output/ys_out/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/hidden_output/ys_out/MatMul_grad/MatMul_1*
_output_shapes

:

�
-train/gradients/hidden_output/2_2D_grad/ShapeConst*!
valueB"2      
   *
dtype0*
_output_shapes
:
�
/train/gradients/hidden_output/2_2D_grad/ReshapeReshapeItrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/control_dependency-train/gradients/hidden_output/2_2D_grad/Shape*
T0*
Tshape0*"
_output_shapes
:2

�
;train/gradients/cell/rnn/transpose_1_grad/InvertPermutationInvertPermutationcell/rnn/concat_2*
T0*
_output_shapes
:
�
3train/gradients/cell/rnn/transpose_1_grad/transpose	Transpose/train/gradients/hidden_output/2_2D_grad/Reshape;train/gradients/cell/rnn/transpose_1_grad/InvertPermutation*
T0*"
_output_shapes
:2
*
Tperm0
�
dtrain/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3cell/rnn/TensorArraycell/rnn/while/Exit_2*'
_class
loc:@cell/rnn/TensorArray*
sourcetrain/gradients*
_output_shapes

:: 
�
`train/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitycell/rnn/while/Exit_2e^train/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*'
_class
loc:@cell/rnn/TensorArray*
_output_shapes
: 
�
jtrain/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3dtrain/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3cell/rnn/TensorArrayStack/range3train/gradients/cell/rnn/transpose_1_grad/transpose`train/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
j
train/gradients/zerosConst*
dtype0*
_output_shapes

:2
*
valueB2
*    
l
train/gradients/zeros_1Const*
valueB2
*    *
dtype0*
_output_shapes

:2

�
1train/gradients/cell/rnn/while/Exit_2_grad/b_exitEnterjtrain/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context
�
1train/gradients/cell/rnn/while/Exit_3_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:2
*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
1train/gradients/cell/rnn/while/Exit_4_grad/b_exitEntertrain/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:2
*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
5train/gradients/cell/rnn/while/Switch_2_grad/b_switchMerge1train/gradients/cell/rnn/while/Exit_2_grad/b_exit<train/gradients/cell/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
�
5train/gradients/cell/rnn/while/Switch_3_grad/b_switchMerge1train/gradients/cell/rnn/while/Exit_3_grad/b_exit<train/gradients/cell/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N* 
_output_shapes
:2
: 
�
5train/gradients/cell/rnn/while/Switch_4_grad/b_switchMerge1train/gradients/cell/rnn/while/Exit_4_grad/b_exit<train/gradients/cell/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N* 
_output_shapes
:2
: 
�
2train/gradients/cell/rnn/while/Merge_2_grad/SwitchSwitch5train/gradients/cell/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 
y
<train/gradients/cell/rnn/while/Merge_2_grad/tuple/group_depsNoOp3^train/gradients/cell/rnn/while/Merge_2_grad/Switch
�
Dtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity2train/gradients/cell/rnn/while/Merge_2_grad/Switch=^train/gradients/cell/rnn/while/Merge_2_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
Ftrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity4train/gradients/cell/rnn/while/Merge_2_grad/Switch:1=^train/gradients/cell/rnn/while/Merge_2_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
2train/gradients/cell/rnn/while/Merge_3_grad/SwitchSwitch5train/gradients/cell/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:2
:2

y
<train/gradients/cell/rnn/while/Merge_3_grad/tuple/group_depsNoOp3^train/gradients/cell/rnn/while/Merge_3_grad/Switch
�
Dtrain/gradients/cell/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity2train/gradients/cell/rnn/while/Merge_3_grad/Switch=^train/gradients/cell/rnn/while/Merge_3_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:2

�
Ftrain/gradients/cell/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity4train/gradients/cell/rnn/while/Merge_3_grad/Switch:1=^train/gradients/cell/rnn/while/Merge_3_grad/tuple/group_deps*
_output_shapes

:2
*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch
�
2train/gradients/cell/rnn/while/Merge_4_grad/SwitchSwitch5train/gradients/cell/rnn/while/Switch_4_grad/b_switchtrain/gradients/b_count_2*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:2
:2

y
<train/gradients/cell/rnn/while/Merge_4_grad/tuple/group_depsNoOp3^train/gradients/cell/rnn/while/Merge_4_grad/Switch
�
Dtrain/gradients/cell/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity2train/gradients/cell/rnn/while/Merge_4_grad/Switch=^train/gradients/cell/rnn/while/Merge_4_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:2

�
Ftrain/gradients/cell/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity4train/gradients/cell/rnn/while/Merge_4_grad/Switch:1=^train/gradients/cell/rnn/while/Merge_4_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:2

�
0train/gradients/cell/rnn/while/Enter_2_grad/ExitExitDtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
0train/gradients/cell/rnn/while/Enter_3_grad/ExitExitDtrain/gradients/cell/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes

:2

�
0train/gradients/cell/rnn/while/Enter_4_grad/ExitExitDtrain/gradients/cell/rnn/while/Merge_4_grad/tuple/control_dependency*
_output_shapes

:2
*
T0
�
itrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3otrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterFtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2*
sourcetrain/gradients*
_output_shapes

:: 
�
otrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntercell/rnn/TensorArray*
is_constant(*
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context*
T0*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations 
�
etrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityFtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1j^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
Ytrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3itrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3dtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2etrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_output_shapes
:���������

�
_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*,
_class"
 loc:@cell/rnn/while/Identity_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*,
_class"
 loc:@cell/rnn/while/Identity_1*

stack_name *
_output_shapes
:
�
_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
etrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Entercell/rnn/while/Identity_1^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
dtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2jtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
�
jtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
`train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggere^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2O^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2I^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2K^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2I^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2K^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2G^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2I^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2M^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2O^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
Xtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpG^train/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1Z^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
`train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityYtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Y^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*l
_classb
`^loc:@train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes

:2

�
btrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityFtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1Y^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
train/gradients/AddNAddNFtrain/gradients/cell/rnn/while/Merge_4_grad/tuple/control_dependency_1`train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_4_grad/b_switch*
N*
_output_shapes

:2

�
=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/MulMultrain/gradients/AddNHtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes

:2

�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*;
_class1
/-loc:@cell/rnn/while/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*;
_class1
/-loc:@cell/rnn/while/basic_lstm_cell/Sigmoid_2
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter(cell/rnn/while/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context*
T0*
is_constant(
�
?train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1Multrain/gradients/AddNJtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
_output_shapes

:2
*
T0
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*8
_class.
,*loc:@cell/rnn/while/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*8
_class.
,*loc:@cell/rnn/while/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterEtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter%cell/rnn/while/basic_lstm_cell/Tanh_1^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterEtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp>^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul@^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/MulK^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul*
_output_shapes

:2

�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity?train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1K^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1*
_output_shapes

:2

�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradJtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*
_output_shapes

:2

�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradHtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2

�
<train/gradients/cell/rnn/while/Switch_2_grad_1/NextIterationNextIterationbtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
train/gradients/AddN_1AddNFtrain/gradients/cell/rnn/while/Merge_3_grad/tuple/control_dependency_1Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes

:2

k
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^train/gradients/AddN_1
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentitytrain/gradients/AddN_1K^train/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
_output_shapes

:2
*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identitytrain/gradients/AddN_1K^train/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:2

�
;train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/MulMulRtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyFtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes

:2

�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*9
_class/
-+loc:@cell/rnn/while/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*9
_class/
-+loc:@cell/rnn/while/basic_lstm_cell/Sigmoid
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/EnterEnterAtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter&cell/rnn/while/basic_lstm_cell/Sigmoid^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Ftrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Ltrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ltrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterAtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1MulRtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyHtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:2

�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*,
_class"
 loc:@cell/rnn/while/Identity_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Const*,
_class"
 loc:@cell/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Entercell/rnn/while/Identity_3^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context*
T0*
is_constant(
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp<^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul>^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1
�
Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity;train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/MulI^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
_output_shapes

:2
*
T0*N
_classD
B@loc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1I^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
_output_shapes

:2
*
T0*P
_classF
DBloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1
�
=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/MulMulTtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes

:2

�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*6
_class,
*(loc:@cell/rnn/while/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Const*
	elem_type0*6
_class,
*(loc:@cell/rnn/while/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter#cell/rnn/while/basic_lstm_cell/Tanh^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:2

�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
?train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1MulTtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:2

�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*;
_class1
/-loc:@cell/rnn/while/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*;
_class1
/-loc:@cell/rnn/while/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterEtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter(cell/rnn/while/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterEtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp>^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul@^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/MulK^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
_output_shapes

:2
*
T0*P
_classF
DBloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity?train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1K^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1*
_output_shapes

:2

�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradFtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2

�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradJtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:2

�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradHtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
_output_shapes

:2
*
T0
�
<train/gradients/cell/rnn/while/Switch_3_grad_1/NextIterationNextIterationPtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0*
_output_shapes

:2

�
=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/ShapeConst^train/gradients/Sub*
valueB"2   
   *
dtype0*
_output_shapes
:
�
?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
�
Mtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgs=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/SumSumGtrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradMtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/ReshapeReshape;train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Sum=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*
_output_shapes

:2

�
=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Sum_1SumGtrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradOtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape_1Reshape=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Sum_1?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOp@^train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/ReshapeB^train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/ReshapeI^train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
_output_shapes

:2
*
T0*R
_classH
FDloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1IdentityAtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape_1I^train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 
�
@train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concatConcatV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradAtrain/gradients/cell/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradPtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyItrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradFtrain/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes

:2(*

Tidx0
�
Ftrain/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad@train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:(
�
Ltrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpH^train/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradA^train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity@train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concatM^train/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes

:2(*
T0*S
_classI
GEloc:@train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat
�
Vtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityGtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradM^train/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes
:(*
T0*Z
_classP
NLloc:@train/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMulMatMulTtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyGtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
T0*
_output_shapes

:2*
transpose_a( *
transpose_b(
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter$cell/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulNtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:(*
transpose_a(*
transpose_b( 
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*8
_class.
,*loc:@cell/rnn/while/basic_lstm_cell/concat*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*8
_class.
,*loc:@cell/rnn/while/basic_lstm_cell/concat*

stack_name *
_output_shapes
:
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterItrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Otrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter%cell/rnn/while/basic_lstm_cell/concat^train/gradients/Add*
T0*
_output_shapes

:2*
swap_memory( 
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:2
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterItrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpB^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMulD^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
�
Strain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityAtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMulL^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul*
_output_shapes

:2
�
Utrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityCtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1L^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
_output_shapes

:(*
T0*V
_classL
JHloc:@train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterGtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:(*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeItrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Otrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:(: 
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchItrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*
T0* 
_output_shapes
:(:(
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddJtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1Vtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:(
�
Otrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationEtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:(
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitHtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:(
�
@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
?train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
>train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/modFloorMod@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Const?train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeShape cell/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeNShapeNLtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Ntrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
N* 
_output_shapes
::*
T0*
out_type0
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/ConstConst*3
_class)
'%loc:@cell/rnn/while/TensorArrayReadV3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const*3
_class)
'%loc:@cell/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/EnterEnterGtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context*
T0*
is_constant(
�
Mtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter cell/rnn/while/TensorArrayReadV3^train/gradients/Add*
T0*'
_output_shapes
:���������
*
swap_memory( 
�
Ltrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Rtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^train/gradients/Sub*'
_output_shapes
:���������
*
	elem_type0
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterGtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1Const*,
_class"
 loc:@cell/rnn/while/Identity_4*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1*
	elem_type0*,
_class"
 loc:@cell/rnn/while/Identity_4*

stack_name *
_output_shapes
:
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterItrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Otrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1cell/rnn/while/Identity_4^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterItrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffset>train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/modAtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeNCtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/SliceSliceStrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyGtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetAtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:������������������
�
Btrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice_1SliceStrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyItrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset:1Ctrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*0
_output_shapes
:������������������*
Index0*
T0
�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOpA^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/SliceC^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice_1
�
Strain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentity@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/SliceL^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*S
_classI
GEloc:@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice*'
_output_shapes
:���������

�
Utrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityBtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice_1L^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:2

�
Ftrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes

:(
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterFtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:(*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeHtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Ntrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:(: 
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchHtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
Dtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/AddAddItrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1Utrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:(*
T0
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationDtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Add*
_output_shapes

:(*
T0
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitGtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
Wtrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3]train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^train/gradients/Sub*9
_class/
-+loc:@cell/rnn/while/TensorArrayReadV3/Enter*
sourcetrain/gradients*
_output_shapes

:: 
�
]train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntercell/rnn/TensorArray_1*
T0*9
_class/
-+loc:@cell/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
_train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterCcell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context*
T0*9
_class/
-+loc:@cell/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
Strain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity_train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1X^train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*9
_class/
-+loc:@cell/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
�
Ytrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Wtrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3dtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Strain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyStrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
�
Ctrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
Etrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterCtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Etrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeEtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Ktrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Dtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchEtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2train/gradients/b_count_2*
T0*
_output_shapes
: : 
�
Atrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddFtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Ytrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
Ktrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationAtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
_output_shapes
: *
T0
�
Etrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitDtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0
�
<train/gradients/cell/rnn/while/Switch_4_grad_1/NextIterationNextIterationUtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2

�
ztrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3cell/rnn/TensorArray_1Etrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *)
_class
loc:@cell/rnn/TensorArray_1*
sourcetrain/gradients
�
vtrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityEtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3{^train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@cell/rnn/TensorArray_1*
_output_shapes
: 
�
ltrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3ztrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3!cell/rnn/TensorArrayUnstack/rangevtrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*4
_output_shapes"
 :������������������

�
itrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpm^train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3F^train/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
qtrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityltrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3j^train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*
_classu
sqloc:@train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*+
_output_shapes
:���������

�
strain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityEtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3j^train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*X
_classN
LJloc:@train/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
�
9train/gradients/cell/rnn/transpose_grad/InvertPermutationInvertPermutationcell/rnn/concat*
T0*
_output_shapes
:
�
1train/gradients/cell/rnn/transpose_grad/transpose	Transposeqtrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency9train/gradients/cell/rnn/transpose_grad/InvertPermutation*+
_output_shapes
:���������
*
Tperm0*
T0
�
,train/gradients/hidden_input/2_3D_grad/ShapeShapehidden_input/ys_in/Add*
_output_shapes
:*
T0*
out_type0
�
.train/gradients/hidden_input/2_3D_grad/ReshapeReshape1train/gradients/cell/rnn/transpose_grad/transpose,train/gradients/hidden_input/2_3D_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
1train/gradients/hidden_input/ys_in/Add_grad/ShapeShapehidden_input/ys_in/MatMul*
T0*
out_type0*
_output_shapes
:
}
3train/gradients/hidden_input/ys_in/Add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
Atrain/gradients/hidden_input/ys_in/Add_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/hidden_input/ys_in/Add_grad/Shape3train/gradients/hidden_input/ys_in/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/train/gradients/hidden_input/ys_in/Add_grad/SumSum.train/gradients/hidden_input/2_3D_grad/ReshapeAtrain/gradients/hidden_input/ys_in/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
3train/gradients/hidden_input/ys_in/Add_grad/ReshapeReshape/train/gradients/hidden_input/ys_in/Add_grad/Sum1train/gradients/hidden_input/ys_in/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
1train/gradients/hidden_input/ys_in/Add_grad/Sum_1Sum.train/gradients/hidden_input/2_3D_grad/ReshapeCtrain/gradients/hidden_input/ys_in/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/hidden_input/ys_in/Add_grad/Reshape_1Reshape1train/gradients/hidden_input/ys_in/Add_grad/Sum_13train/gradients/hidden_input/ys_in/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

�
<train/gradients/hidden_input/ys_in/Add_grad/tuple/group_depsNoOp4^train/gradients/hidden_input/ys_in/Add_grad/Reshape6^train/gradients/hidden_input/ys_in/Add_grad/Reshape_1
�
Dtrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependencyIdentity3train/gradients/hidden_input/ys_in/Add_grad/Reshape=^train/gradients/hidden_input/ys_in/Add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/hidden_input/ys_in/Add_grad/Reshape*'
_output_shapes
:���������

�
Ftrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependency_1Identity5train/gradients/hidden_input/ys_in/Add_grad/Reshape_1=^train/gradients/hidden_input/ys_in/Add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/hidden_input/ys_in/Add_grad/Reshape_1*
_output_shapes
:

�
5train/gradients/hidden_input/ys_in/MatMul_grad/MatMulMatMulDtrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependencyhidden_input/Weights/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
7train/gradients/hidden_input/ys_in/MatMul_grad/MatMul_1MatMulhidden_input/2_2DDtrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
?train/gradients/hidden_input/ys_in/MatMul_grad/tuple/group_depsNoOp6^train/gradients/hidden_input/ys_in/MatMul_grad/MatMul8^train/gradients/hidden_input/ys_in/MatMul_grad/MatMul_1
�
Gtrain/gradients/hidden_input/ys_in/MatMul_grad/tuple/control_dependencyIdentity5train/gradients/hidden_input/ys_in/MatMul_grad/MatMul@^train/gradients/hidden_input/ys_in/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/hidden_input/ys_in/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Itrain/gradients/hidden_input/ys_in/MatMul_grad/tuple/control_dependency_1Identity7train/gradients/hidden_input/ys_in/MatMul_grad/MatMul_1@^train/gradients/hidden_input/ys_in/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@train/gradients/hidden_input/ys_in/MatMul_grad/MatMul_1*
_output_shapes

:

�
train/beta1_power/initial_valueConst*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
train/beta1_power/readIdentitytrain/beta1_power*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
: 
�
train/beta2_power/readIdentitytrain/beta2_power*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
: 
�
+hidden_input/Weights/Adam/Initializer/zerosConst*'
_class
loc:@hidden_input/Weights*
valueB
*    *
dtype0*
_output_shapes

:

�
hidden_input/Weights/Adam
VariableV2*
shared_name *'
_class
loc:@hidden_input/Weights*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
 hidden_input/Weights/Adam/AssignAssignhidden_input/Weights/Adam+hidden_input/Weights/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@hidden_input/Weights*
validate_shape(*
_output_shapes

:

�
hidden_input/Weights/Adam/readIdentityhidden_input/Weights/Adam*
T0*'
_class
loc:@hidden_input/Weights*
_output_shapes

:

�
-hidden_input/Weights/Adam_1/Initializer/zerosConst*'
_class
loc:@hidden_input/Weights*
valueB
*    *
dtype0*
_output_shapes

:

�
hidden_input/Weights/Adam_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *'
_class
loc:@hidden_input/Weights*
	container *
shape
:

�
"hidden_input/Weights/Adam_1/AssignAssignhidden_input/Weights/Adam_1-hidden_input/Weights/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@hidden_input/Weights*
validate_shape(*
_output_shapes

:

�
 hidden_input/Weights/Adam_1/readIdentityhidden_input/Weights/Adam_1*
T0*'
_class
loc:@hidden_input/Weights*
_output_shapes

:

�
*hidden_input/biases/Adam/Initializer/zerosConst*&
_class
loc:@hidden_input/biases*
valueB
*    *
dtype0*
_output_shapes
:

�
hidden_input/biases/Adam
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *&
_class
loc:@hidden_input/biases*
	container *
shape:

�
hidden_input/biases/Adam/AssignAssignhidden_input/biases/Adam*hidden_input/biases/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*&
_class
loc:@hidden_input/biases
�
hidden_input/biases/Adam/readIdentityhidden_input/biases/Adam*
T0*&
_class
loc:@hidden_input/biases*
_output_shapes
:

�
,hidden_input/biases/Adam_1/Initializer/zerosConst*&
_class
loc:@hidden_input/biases*
valueB
*    *
dtype0*
_output_shapes
:

�
hidden_input/biases/Adam_1
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *&
_class
loc:@hidden_input/biases*
	container *
shape:

�
!hidden_input/biases/Adam_1/AssignAssignhidden_input/biases/Adam_1,hidden_input/biases/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*&
_class
loc:@hidden_input/biases
�
hidden_input/biases/Adam_1/readIdentityhidden_input/biases/Adam_1*
T0*&
_class
loc:@hidden_input/biases*
_output_shapes
:

�
6cell/rnn/basic_lstm_cell/kernel/Adam/Initializer/zerosConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB(*    *
dtype0*
_output_shapes

:(
�
$cell/rnn/basic_lstm_cell/kernel/Adam
VariableV2*
	container *
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
+cell/rnn/basic_lstm_cell/kernel/Adam/AssignAssign$cell/rnn/basic_lstm_cell/kernel/Adam6cell/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
)cell/rnn/basic_lstm_cell/kernel/Adam/readIdentity$cell/rnn/basic_lstm_cell/kernel/Adam*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
8cell/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB(*    *
dtype0*
_output_shapes

:(
�
&cell/rnn/basic_lstm_cell/kernel/Adam_1
VariableV2*
	container *
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
-cell/rnn/basic_lstm_cell/kernel/Adam_1/AssignAssign&cell/rnn/basic_lstm_cell/kernel/Adam_18cell/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
+cell/rnn/basic_lstm_cell/kernel/Adam_1/readIdentity&cell/rnn/basic_lstm_cell/kernel/Adam_1*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
4cell/rnn/basic_lstm_cell/bias/Adam/Initializer/zerosConst*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
"cell/rnn/basic_lstm_cell/bias/Adam
VariableV2*
	container *
shape:(*
dtype0*
_output_shapes
:(*
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias
�
)cell/rnn/basic_lstm_cell/bias/Adam/AssignAssign"cell/rnn/basic_lstm_cell/bias/Adam4cell/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
'cell/rnn/basic_lstm_cell/bias/Adam/readIdentity"cell/rnn/basic_lstm_cell/bias/Adam*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
:(
�
6cell/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:(*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB(*    
�
$cell/rnn/basic_lstm_cell/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
	container *
shape:(
�
+cell/rnn/basic_lstm_cell/bias/Adam_1/AssignAssign$cell/rnn/basic_lstm_cell/bias/Adam_16cell/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
)cell/rnn/basic_lstm_cell/bias/Adam_1/readIdentity$cell/rnn/basic_lstm_cell/bias/Adam_1*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
:(
�
,hidden_output/Weights/Adam/Initializer/zerosConst*(
_class
loc:@hidden_output/Weights*
valueB
*    *
dtype0*
_output_shapes

:

�
hidden_output/Weights/Adam
VariableV2*
shared_name *(
_class
loc:@hidden_output/Weights*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
!hidden_output/Weights/Adam/AssignAssignhidden_output/Weights/Adam,hidden_output/Weights/Adam/Initializer/zeros*
T0*(
_class
loc:@hidden_output/Weights*
validate_shape(*
_output_shapes

:
*
use_locking(
�
hidden_output/Weights/Adam/readIdentityhidden_output/Weights/Adam*
T0*(
_class
loc:@hidden_output/Weights*
_output_shapes

:

�
.hidden_output/Weights/Adam_1/Initializer/zerosConst*(
_class
loc:@hidden_output/Weights*
valueB
*    *
dtype0*
_output_shapes

:

�
hidden_output/Weights/Adam_1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *(
_class
loc:@hidden_output/Weights*
	container 
�
#hidden_output/Weights/Adam_1/AssignAssignhidden_output/Weights/Adam_1.hidden_output/Weights/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*(
_class
loc:@hidden_output/Weights
�
!hidden_output/Weights/Adam_1/readIdentityhidden_output/Weights/Adam_1*
T0*(
_class
loc:@hidden_output/Weights*
_output_shapes

:

�
+hidden_output/biases/Adam/Initializer/zerosConst*'
_class
loc:@hidden_output/biases*
valueB*    *
dtype0*
_output_shapes
:
�
hidden_output/biases/Adam
VariableV2*
shared_name *'
_class
loc:@hidden_output/biases*
	container *
shape:*
dtype0*
_output_shapes
:
�
 hidden_output/biases/Adam/AssignAssignhidden_output/biases/Adam+hidden_output/biases/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@hidden_output/biases*
validate_shape(*
_output_shapes
:
�
hidden_output/biases/Adam/readIdentityhidden_output/biases/Adam*
T0*'
_class
loc:@hidden_output/biases*
_output_shapes
:
�
-hidden_output/biases/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*'
_class
loc:@hidden_output/biases*
valueB*    
�
hidden_output/biases/Adam_1
VariableV2*'
_class
loc:@hidden_output/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
"hidden_output/biases/Adam_1/AssignAssignhidden_output/biases/Adam_1-hidden_output/biases/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@hidden_output/biases*
validate_shape(*
_output_shapes
:
�
 hidden_output/biases/Adam_1/readIdentityhidden_output/biases/Adam_1*
T0*'
_class
loc:@hidden_output/biases*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *���;*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
0train/Adam/update_hidden_input/Weights/ApplyAdam	ApplyAdamhidden_input/Weightshidden_input/Weights/Adamhidden_input/Weights/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonItrain/gradients/hidden_input/ys_in/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:
*
use_locking( *
T0*'
_class
loc:@hidden_input/Weights
�
/train/Adam/update_hidden_input/biases/ApplyAdam	ApplyAdamhidden_input/biaseshidden_input/biases/Adamhidden_input/biases/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonFtrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependency_1*
T0*&
_class
loc:@hidden_input/biases*
use_nesterov( *
_output_shapes
:
*
use_locking( 
�
;train/Adam/update_cell/rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamcell/rnn/basic_lstm_cell/kernel$cell/rnn/basic_lstm_cell/kernel/Adam&cell/rnn/basic_lstm_cell/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonHtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_nesterov( *
_output_shapes

:(*
use_locking( *
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
9train/Adam/update_cell/rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamcell/rnn/basic_lstm_cell/bias"cell/rnn/basic_lstm_cell/bias/Adam$cell/rnn/basic_lstm_cell/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonItrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes
:(
�
1train/Adam/update_hidden_output/Weights/ApplyAdam	ApplyAdamhidden_output/Weightshidden_output/Weights/Adamhidden_output/Weights/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonKtrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/control_dependency_1*
T0*(
_class
loc:@hidden_output/Weights*
use_nesterov( *
_output_shapes

:
*
use_locking( 
�
0train/Adam/update_hidden_output/biases/ApplyAdam	ApplyAdamhidden_output/biaseshidden_output/biases/Adamhidden_output/biases/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonHtrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@hidden_output/biases*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1:^train/Adam/update_cell/rnn/basic_lstm_cell/bias/ApplyAdam<^train/Adam/update_cell/rnn/basic_lstm_cell/kernel/ApplyAdam1^train/Adam/update_hidden_input/Weights/ApplyAdam0^train/Adam/update_hidden_input/biases/ApplyAdam2^train/Adam/update_hidden_output/Weights/ApplyAdam1^train/Adam/update_hidden_output/biases/ApplyAdam*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2:^train/Adam/update_cell/rnn/basic_lstm_cell/bias/ApplyAdam<^train/Adam/update_cell/rnn/basic_lstm_cell/kernel/ApplyAdam1^train/Adam/update_hidden_input/Weights/ApplyAdam0^train/Adam/update_hidden_input/biases/ApplyAdam2^train/Adam/update_hidden_output/Weights/ApplyAdam1^train/Adam/update_hidden_output/biases/ApplyAdam*
_output_shapes
: *
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
: 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1:^train/Adam/update_cell/rnn/basic_lstm_cell/bias/ApplyAdam<^train/Adam/update_cell/rnn/basic_lstm_cell/kernel/ApplyAdam1^train/Adam/update_hidden_input/Weights/ApplyAdam0^train/Adam/update_hidden_input/biases/ApplyAdam2^train/Adam/update_hidden_output/Weights/ApplyAdam1^train/Adam/update_hidden_output/biases/ApplyAdam
[
Merge/MergeSummaryMergeSummaryloss/average_loss/loss*
N*
_output_shapes
: "<�J],     ئ��	��Q1(�AJ��
�+�+
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z
�
!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
1
Square
x"T
y"T"
Ttype:

2	
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype�
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( �
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring �
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:�
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring�
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype�
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype�
9
TensorArraySizeV3

handle
flow_in
size�
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring �
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype�
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.9.02v1.9.0-0-g25c197e023��
t
	inputs/xsPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
t
	inputs/ysPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
h
hidden_input/2_2D/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
hidden_input/2_2DReshape	inputs/xshidden_input/2_2D/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
4hidden_input/Weights/Initializer/random_normal/shapeConst*'
_class
loc:@hidden_input/Weights*
valueB"   
   *
dtype0*
_output_shapes
:
�
3hidden_input/Weights/Initializer/random_normal/meanConst*'
_class
loc:@hidden_input/Weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
5hidden_input/Weights/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *'
_class
loc:@hidden_input/Weights*
valueB
 *  �?
�
Chidden_input/Weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal4hidden_input/Weights/Initializer/random_normal/shape*
dtype0*
_output_shapes

:
*

seed *
T0*'
_class
loc:@hidden_input/Weights*
seed2 
�
2hidden_input/Weights/Initializer/random_normal/mulMulChidden_input/Weights/Initializer/random_normal/RandomStandardNormal5hidden_input/Weights/Initializer/random_normal/stddev*
T0*'
_class
loc:@hidden_input/Weights*
_output_shapes

:

�
.hidden_input/Weights/Initializer/random_normalAdd2hidden_input/Weights/Initializer/random_normal/mul3hidden_input/Weights/Initializer/random_normal/mean*
T0*'
_class
loc:@hidden_input/Weights*
_output_shapes

:

�
hidden_input/Weights
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *'
_class
loc:@hidden_input/Weights*
	container 
�
hidden_input/Weights/AssignAssignhidden_input/Weights.hidden_input/Weights/Initializer/random_normal*
T0*'
_class
loc:@hidden_input/Weights*
validate_shape(*
_output_shapes

:
*
use_locking(
�
hidden_input/Weights/readIdentityhidden_input/Weights*
T0*'
_class
loc:@hidden_input/Weights*
_output_shapes

:

�
%hidden_input/biases/Initializer/ConstConst*&
_class
loc:@hidden_input/biases*
valueB
*���=*
dtype0*
_output_shapes
:

�
hidden_input/biases
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *&
_class
loc:@hidden_input/biases*
	container *
shape:

�
hidden_input/biases/AssignAssignhidden_input/biases%hidden_input/biases/Initializer/Const*
T0*&
_class
loc:@hidden_input/biases*
validate_shape(*
_output_shapes
:
*
use_locking(
�
hidden_input/biases/readIdentityhidden_input/biases*
T0*&
_class
loc:@hidden_input/biases*
_output_shapes
:

�
hidden_input/ys_in/MatMulMatMulhidden_input/2_2Dhidden_input/Weights/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
hidden_input/ys_in/AddAddhidden_input/ys_in/MatMulhidden_input/biases/read*
T0*'
_output_shapes
:���������

l
hidden_input/2_3D/shapeConst*
dtype0*
_output_shapes
:*!
valueB"����   
   
�
hidden_input/2_3DReshapehidden_input/ys_in/Addhidden_input/2_3D/shape*
T0*
Tshape0*+
_output_shapes
:���������

y
/cell/initial_state/BasicLSTMCellZeroState/ConstConst*
valueB:2*
dtype0*
_output_shapes
:
{
1cell/initial_state/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
w
5cell/initial_state/BasicLSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
0cell/initial_state/BasicLSTMCellZeroState/concatConcatV2/cell/initial_state/BasicLSTMCellZeroState/Const1cell/initial_state/BasicLSTMCellZeroState/Const_15cell/initial_state/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
z
5cell/initial_state/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/cell/initial_state/BasicLSTMCellZeroState/zerosFill0cell/initial_state/BasicLSTMCellZeroState/concat5cell/initial_state/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:2
*
T0*

index_type0
{
1cell/initial_state/BasicLSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:2
{
1cell/initial_state/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
{
1cell/initial_state/BasicLSTMCellZeroState/Const_4Const*
valueB:2*
dtype0*
_output_shapes
:
{
1cell/initial_state/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
y
7cell/initial_state/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
2cell/initial_state/BasicLSTMCellZeroState/concat_1ConcatV21cell/initial_state/BasicLSTMCellZeroState/Const_41cell/initial_state/BasicLSTMCellZeroState/Const_57cell/initial_state/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
|
7cell/initial_state/BasicLSTMCellZeroState/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
1cell/initial_state/BasicLSTMCellZeroState/zeros_1Fill2cell/initial_state/BasicLSTMCellZeroState/concat_17cell/initial_state/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes

:2

{
1cell/initial_state/BasicLSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:2
{
1cell/initial_state/BasicLSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:

O
cell/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
V
cell/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
V
cell/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
z
cell/rnn/rangeRangecell/rnn/range/startcell/rnn/Rankcell/rnn/range/delta*
_output_shapes
:*

Tidx0
i
cell/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
V
cell/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/concatConcatV2cell/rnn/concat/values_0cell/rnn/rangecell/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
cell/rnn/transpose	Transposehidden_input/2_3Dcell/rnn/concat*
T0*+
_output_shapes
:���������
*
Tperm0
`
cell/rnn/ShapeShapecell/rnn/transpose*
T0*
out_type0*
_output_shapes
:
f
cell/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
h
cell/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
cell/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
cell/rnn/strided_sliceStridedSlicecell/rnn/Shapecell/rnn/strided_slice/stackcell/rnn/strided_slice/stack_1cell/rnn/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
b
cell/rnn/Shape_1Shapecell/rnn/transpose*
T0*
out_type0*
_output_shapes
:
h
cell/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
j
 cell/rnn/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
j
 cell/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
cell/rnn/strided_slice_1StridedSlicecell/rnn/Shape_1cell/rnn/strided_slice_1/stack cell/rnn/strided_slice_1/stack_1 cell/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
b
cell/rnn/Shape_2Shapecell/rnn/transpose*
_output_shapes
:*
T0*
out_type0
h
cell/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
j
 cell/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
j
 cell/rnn/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
cell/rnn/strided_slice_2StridedSlicecell/rnn/Shape_2cell/rnn/strided_slice_2/stack cell/rnn/strided_slice_2/stack_1 cell/rnn/strided_slice_2/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
Y
cell/rnn/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
cell/rnn/ExpandDims
ExpandDimscell/rnn/strided_slice_2cell/rnn/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
X
cell/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB:

X
cell/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/concat_1ConcatV2cell/rnn/ExpandDimscell/rnn/Constcell/rnn/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
Y
cell/rnn/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
cell/rnn/zerosFillcell/rnn/concat_1cell/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������

O
cell/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/TensorArrayTensorArrayV3cell/rnn/strided_slice_1*4
tensor_array_namecell/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
�
cell/rnn/TensorArray_1TensorArrayV3cell/rnn/strided_slice_1*$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*3
tensor_array_namecell/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
s
!cell/rnn/TensorArrayUnstack/ShapeShapecell/rnn/transpose*
_output_shapes
:*
T0*
out_type0
y
/cell/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
{
1cell/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
{
1cell/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
)cell/rnn/TensorArrayUnstack/strided_sliceStridedSlice!cell/rnn/TensorArrayUnstack/Shape/cell/rnn/TensorArrayUnstack/strided_slice/stack1cell/rnn/TensorArrayUnstack/strided_slice/stack_11cell/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
i
'cell/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
i
'cell/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
!cell/rnn/TensorArrayUnstack/rangeRange'cell/rnn/TensorArrayUnstack/range/start)cell/rnn/TensorArrayUnstack/strided_slice'cell/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Ccell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3cell/rnn/TensorArray_1!cell/rnn/TensorArrayUnstack/rangecell/rnn/transposecell/rnn/TensorArray_1:1*
T0*%
_class
loc:@cell/rnn/transpose*
_output_shapes
: 
T
cell/rnn/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :
j
cell/rnn/MaximumMaximumcell/rnn/Maximum/xcell/rnn/strided_slice_1*
T0*
_output_shapes
: 
h
cell/rnn/MinimumMinimumcell/rnn/strided_slice_1cell/rnn/Maximum*
T0*
_output_shapes
: 
b
 cell/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/while/EnterEnter cell/rnn/while/iteration_counter*
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context*
T0*
is_constant( 
�
cell/rnn/while/Enter_1Entercell/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
cell/rnn/while/Enter_2Entercell/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
cell/rnn/while/Enter_3Enter/cell/initial_state/BasicLSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes

:2
*,

frame_namecell/rnn/while/while_context*
T0*
is_constant( 
�
cell/rnn/while/Enter_4Enter1cell/initial_state/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:2
*,

frame_namecell/rnn/while/while_context
}
cell/rnn/while/MergeMergecell/rnn/while/Entercell/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
�
cell/rnn/while/Merge_1Mergecell/rnn/while/Enter_1cell/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
cell/rnn/while/Merge_2Mergecell/rnn/while/Enter_2cell/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
cell/rnn/while/Merge_3Mergecell/rnn/while/Enter_3cell/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:2
: 
�
cell/rnn/while/Merge_4Mergecell/rnn/while/Enter_4cell/rnn/while/NextIteration_4*
T0*
N* 
_output_shapes
:2
: 
m
cell/rnn/while/LessLesscell/rnn/while/Mergecell/rnn/while/Less/Enter*
_output_shapes
: *
T0
�
cell/rnn/while/Less/EnterEntercell/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
s
cell/rnn/while/Less_1Lesscell/rnn/while/Merge_1cell/rnn/while/Less_1/Enter*
_output_shapes
: *
T0
�
cell/rnn/while/Less_1/EnterEntercell/rnn/Minimum*
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context*
T0*
is_constant(
k
cell/rnn/while/LogicalAnd
LogicalAndcell/rnn/while/Lesscell/rnn/while/Less_1*
_output_shapes
: 
V
cell/rnn/while/LoopCondLoopCondcell/rnn/while/LogicalAnd*
_output_shapes
: 
�
cell/rnn/while/SwitchSwitchcell/rnn/while/Mergecell/rnn/while/LoopCond*
T0*'
_class
loc:@cell/rnn/while/Merge*
_output_shapes
: : 
�
cell/rnn/while/Switch_1Switchcell/rnn/while/Merge_1cell/rnn/while/LoopCond*
T0*)
_class
loc:@cell/rnn/while/Merge_1*
_output_shapes
: : 
�
cell/rnn/while/Switch_2Switchcell/rnn/while/Merge_2cell/rnn/while/LoopCond*
T0*)
_class
loc:@cell/rnn/while/Merge_2*
_output_shapes
: : 
�
cell/rnn/while/Switch_3Switchcell/rnn/while/Merge_3cell/rnn/while/LoopCond*
T0*)
_class
loc:@cell/rnn/while/Merge_3*(
_output_shapes
:2
:2

�
cell/rnn/while/Switch_4Switchcell/rnn/while/Merge_4cell/rnn/while/LoopCond*
T0*)
_class
loc:@cell/rnn/while/Merge_4*(
_output_shapes
:2
:2

]
cell/rnn/while/IdentityIdentitycell/rnn/while/Switch:1*
T0*
_output_shapes
: 
a
cell/rnn/while/Identity_1Identitycell/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
a
cell/rnn/while/Identity_2Identitycell/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
i
cell/rnn/while/Identity_3Identitycell/rnn/while/Switch_3:1*
T0*
_output_shapes

:2

i
cell/rnn/while/Identity_4Identitycell/rnn/while/Switch_4:1*
_output_shapes

:2
*
T0
p
cell/rnn/while/add/yConst^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
cell/rnn/while/addAddcell/rnn/while/Identitycell/rnn/while/add/y*
T0*
_output_shapes
: 
�
 cell/rnn/while/TensorArrayReadV3TensorArrayReadV3&cell/rnn/while/TensorArrayReadV3/Entercell/rnn/while/Identity_1(cell/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
&cell/rnn/while/TensorArrayReadV3/EnterEntercell/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
(cell/rnn/while/TensorArrayReadV3/Enter_1EnterCcell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
@cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB
 *�衾*
dtype0*
_output_shapes
: 
�
>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
Hcell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform@cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:(*

seed *
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
seed2 
�
>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSub>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/max>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
_output_shapes
: 
�
>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulHcell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
:cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAdd>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mul>cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
cell/rnn/basic_lstm_cell/kernel
VariableV2*
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
	container 
�
&cell/rnn/basic_lstm_cell/kernel/AssignAssigncell/rnn/basic_lstm_cell/kernel:cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
z
$cell/rnn/basic_lstm_cell/kernel/readIdentitycell/rnn/basic_lstm_cell/kernel*
_output_shapes

:(*
T0
�
/cell/rnn/basic_lstm_cell/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:(*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB(*    
�
cell/rnn/basic_lstm_cell/bias
VariableV2*
	container *
shape:(*
dtype0*
_output_shapes
:(*
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias
�
$cell/rnn/basic_lstm_cell/bias/AssignAssigncell/rnn/basic_lstm_cell/bias/cell/rnn/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
r
"cell/rnn/basic_lstm_cell/bias/readIdentitycell/rnn/basic_lstm_cell/bias*
T0*
_output_shapes
:(
�
$cell/rnn/while/basic_lstm_cell/ConstConst^cell/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
*cell/rnn/while/basic_lstm_cell/concat/axisConst^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
%cell/rnn/while/basic_lstm_cell/concatConcatV2 cell/rnn/while/TensorArrayReadV3cell/rnn/while/Identity_4*cell/rnn/while/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:2
�
%cell/rnn/while/basic_lstm_cell/MatMulMatMul%cell/rnn/while/basic_lstm_cell/concat+cell/rnn/while/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes

:2(*
transpose_a( *
transpose_b( 
�
+cell/rnn/while/basic_lstm_cell/MatMul/EnterEnter$cell/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*,

frame_namecell/rnn/while/while_context
�
&cell/rnn/while/basic_lstm_cell/BiasAddBiasAdd%cell/rnn/while/basic_lstm_cell/MatMul,cell/rnn/while/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:2(
�
,cell/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter"cell/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*,

frame_namecell/rnn/while/while_context
�
&cell/rnn/while/basic_lstm_cell/Const_1Const^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
$cell/rnn/while/basic_lstm_cell/splitSplit$cell/rnn/while/basic_lstm_cell/Const&cell/rnn/while/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:2
:2
:2
:2
*
	num_split
�
&cell/rnn/while/basic_lstm_cell/Const_2Const^cell/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
"cell/rnn/while/basic_lstm_cell/AddAdd&cell/rnn/while/basic_lstm_cell/split:2&cell/rnn/while/basic_lstm_cell/Const_2*
T0*
_output_shapes

:2

~
&cell/rnn/while/basic_lstm_cell/SigmoidSigmoid"cell/rnn/while/basic_lstm_cell/Add*
T0*
_output_shapes

:2

�
"cell/rnn/while/basic_lstm_cell/MulMulcell/rnn/while/Identity_3&cell/rnn/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:2

�
(cell/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid$cell/rnn/while/basic_lstm_cell/split*
T0*
_output_shapes

:2

|
#cell/rnn/while/basic_lstm_cell/TanhTanh&cell/rnn/while/basic_lstm_cell/split:1*
_output_shapes

:2
*
T0
�
$cell/rnn/while/basic_lstm_cell/Mul_1Mul(cell/rnn/while/basic_lstm_cell/Sigmoid_1#cell/rnn/while/basic_lstm_cell/Tanh*
T0*
_output_shapes

:2

�
$cell/rnn/while/basic_lstm_cell/Add_1Add"cell/rnn/while/basic_lstm_cell/Mul$cell/rnn/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes

:2

|
%cell/rnn/while/basic_lstm_cell/Tanh_1Tanh$cell/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:2

�
(cell/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid&cell/rnn/while/basic_lstm_cell/split:3*
_output_shapes

:2
*
T0
�
$cell/rnn/while/basic_lstm_cell/Mul_2Mul%cell/rnn/while/basic_lstm_cell/Tanh_1(cell/rnn/while/basic_lstm_cell/Sigmoid_2*
_output_shapes

:2
*
T0
�
2cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV38cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entercell/rnn/while/Identity_1$cell/rnn/while/basic_lstm_cell/Mul_2cell/rnn/while/Identity_2*
T0*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
8cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntercell/rnn/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:*,

frame_namecell/rnn/while/while_context*
T0*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2
r
cell/rnn/while/add_1/yConst^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
cell/rnn/while/add_1Addcell/rnn/while/Identity_1cell/rnn/while/add_1/y*
_output_shapes
: *
T0
b
cell/rnn/while/NextIterationNextIterationcell/rnn/while/add*
_output_shapes
: *
T0
f
cell/rnn/while/NextIteration_1NextIterationcell/rnn/while/add_1*
T0*
_output_shapes
: 
�
cell/rnn/while/NextIteration_2NextIteration2cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
~
cell/rnn/while/NextIteration_3NextIteration$cell/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:2

~
cell/rnn/while/NextIteration_4NextIteration$cell/rnn/while/basic_lstm_cell/Mul_2*
T0*
_output_shapes

:2

S
cell/rnn/while/ExitExitcell/rnn/while/Switch*
T0*
_output_shapes
: 
W
cell/rnn/while/Exit_1Exitcell/rnn/while/Switch_1*
_output_shapes
: *
T0
W
cell/rnn/while/Exit_2Exitcell/rnn/while/Switch_2*
_output_shapes
: *
T0
_
cell/rnn/while/Exit_3Exitcell/rnn/while/Switch_3*
_output_shapes

:2
*
T0
_
cell/rnn/while/Exit_4Exitcell/rnn/while/Switch_4*
T0*
_output_shapes

:2

�
+cell/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3cell/rnn/TensorArraycell/rnn/while/Exit_2*'
_class
loc:@cell/rnn/TensorArray*
_output_shapes
: 
�
%cell/rnn/TensorArrayStack/range/startConst*'
_class
loc:@cell/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
%cell/rnn/TensorArrayStack/range/deltaConst*'
_class
loc:@cell/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
�
cell/rnn/TensorArrayStack/rangeRange%cell/rnn/TensorArrayStack/range/start+cell/rnn/TensorArrayStack/TensorArraySizeV3%cell/rnn/TensorArrayStack/range/delta*'
_class
loc:@cell/rnn/TensorArray*#
_output_shapes
:���������*

Tidx0
�
-cell/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3cell/rnn/TensorArraycell/rnn/TensorArrayStack/rangecell/rnn/while/Exit_2*
element_shape
:2
*'
_class
loc:@cell/rnn/TensorArray*
dtype0*"
_output_shapes
:2

Z
cell/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
Q
cell/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
X
cell/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
X
cell/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
cell/rnn/range_1Rangecell/rnn/range_1/startcell/rnn/Rank_1cell/rnn/range_1/delta*
_output_shapes
:*

Tidx0
k
cell/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
X
cell/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cell/rnn/concat_2ConcatV2cell/rnn/concat_2/values_0cell/rnn/range_1cell/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
cell/rnn/transpose_1	Transpose-cell/rnn/TensorArrayStack/TensorArrayGatherV3cell/rnn/concat_2*"
_output_shapes
:2
*
Tperm0*
T0
i
hidden_output/2_2D/shapeConst*
valueB"����
   *
dtype0*
_output_shapes
:
�
hidden_output/2_2DReshapecell/rnn/transpose_1hidden_output/2_2D/shape*
_output_shapes
:	�
*
T0*
Tshape0
�
5hidden_output/Weights/Initializer/random_normal/shapeConst*(
_class
loc:@hidden_output/Weights*
valueB"
      *
dtype0*
_output_shapes
:
�
4hidden_output/Weights/Initializer/random_normal/meanConst*(
_class
loc:@hidden_output/Weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6hidden_output/Weights/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *(
_class
loc:@hidden_output/Weights*
valueB
 *  �?
�
Dhidden_output/Weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal5hidden_output/Weights/Initializer/random_normal/shape*
T0*(
_class
loc:@hidden_output/Weights*
seed2 *
dtype0*
_output_shapes

:
*

seed 
�
3hidden_output/Weights/Initializer/random_normal/mulMulDhidden_output/Weights/Initializer/random_normal/RandomStandardNormal6hidden_output/Weights/Initializer/random_normal/stddev*
T0*(
_class
loc:@hidden_output/Weights*
_output_shapes

:

�
/hidden_output/Weights/Initializer/random_normalAdd3hidden_output/Weights/Initializer/random_normal/mul4hidden_output/Weights/Initializer/random_normal/mean*
_output_shapes

:
*
T0*(
_class
loc:@hidden_output/Weights
�
hidden_output/Weights
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *(
_class
loc:@hidden_output/Weights*
	container *
shape
:

�
hidden_output/Weights/AssignAssignhidden_output/Weights/hidden_output/Weights/Initializer/random_normal*
use_locking(*
T0*(
_class
loc:@hidden_output/Weights*
validate_shape(*
_output_shapes

:

�
hidden_output/Weights/readIdentityhidden_output/Weights*
T0*(
_class
loc:@hidden_output/Weights*
_output_shapes

:

�
&hidden_output/biases/Initializer/ConstConst*'
_class
loc:@hidden_output/biases*
valueB*���=*
dtype0*
_output_shapes
:
�
hidden_output/biases
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@hidden_output/biases*
	container *
shape:
�
hidden_output/biases/AssignAssignhidden_output/biases&hidden_output/biases/Initializer/Const*
use_locking(*
T0*'
_class
loc:@hidden_output/biases*
validate_shape(*
_output_shapes
:
�
hidden_output/biases/readIdentityhidden_output/biases*
T0*'
_class
loc:@hidden_output/biases*
_output_shapes
:
�
hidden_output/ys_out/MatMulMatMulhidden_output/2_2Dhidden_output/Weights/read*
T0*
_output_shapes
:	�*
transpose_a( *
transpose_b( 
�
hidden_output/ys_out/AddAddhidden_output/ys_out/MatMulhidden_output/biases/read*
T0*
_output_shapes
:	�
p
loss/reshape_prediction/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
loss/reshape_predictionReshapehidden_output/ys_out/Addloss/reshape_prediction/shape*
T0*
Tshape0*
_output_shapes	
:�
l
loss/reshape_target/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
loss/reshape_targetReshape	inputs/ysloss/reshape_target/shape*
T0*
Tshape0*#
_output_shapes
:���������
d
loss/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:�
T
loss/ones/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
	loss/onesFillloss/ones/shape_as_tensorloss/ones/Const*
T0*

index_type0*
_output_shapes	
:�
j
loss/losses/SubSubloss/reshape_targetloss/reshape_prediction*
_output_shapes	
:�*
T0
S
loss/losses/SquareSquareloss/losses/Sub*
T0*
_output_shapes	
:�
[
loss/losses/mulMulloss/losses/Square	loss/ones*
T0*
_output_shapes	
:�
V
loss/losses/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
Z
loss/losses/addAdd	loss/onesloss/losses/add/y*
T0*
_output_shapes	
:�
f
loss/losses/truedivRealDivloss/losses/mulloss/losses/add*
_output_shapes	
:�*
T0
a
loss/average_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/average_loss/sum_lossSumloss/losses/truedivloss/average_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
 loss/average_loss/average_loss/yConst*
valueB
 *  HB*
dtype0*
_output_shapes
: 
�
loss/average_loss/average_lossRealDivloss/average_loss/sum_loss loss/average_loss/average_loss/y*
T0*
_output_shapes
: 
r
loss/average_loss/loss/tagsConst*
dtype0*
_output_shapes
: *'
valueB Bloss/average_loss/loss
�
loss/average_loss/lossScalarSummaryloss/average_loss/loss/tagsloss/average_loss/average_loss*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Y
train/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
train/gradients/f_count_1Entertrain/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *,

frame_namecell/rnn/while/while_context
�
train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
N*
_output_shapes
: : *
T0
s
train/gradients/SwitchSwitchtrain/gradients/Mergecell/rnn/while/LoopCond*
T0*
_output_shapes
: : 
q
train/gradients/Add/yConst^cell/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
T0*
_output_shapes
: 
�
train/gradients/NextIterationNextIterationtrain/gradients/Addf^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2P^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2J^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2L^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2J^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2L^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2H^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2J^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2N^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2P^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
T0*
_output_shapes
: 
Z
train/gradients/f_count_2Exittrain/gradients/Switch*
_output_shapes
: *
T0
Y
train/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context
�
train/gradients/Merge_1Mergetrain/gradients/b_count_1train/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
train/gradients/GreaterEqualGreaterEqualtrain/gradients/Merge_1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
"train/gradients/GreaterEqual/EnterEntertrain/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context
[
train/gradients/b_count_2LoopCondtrain/gradients/GreaterEqual*
_output_shapes
: 
y
train/gradients/Switch_1Switchtrain/gradients/Merge_1train/gradients/b_count_2*
_output_shapes
: : *
T0
{
train/gradients/SubSubtrain/gradients/Switch_1:1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
train/gradients/NextIteration_1NextIterationtrain/gradients/Suba^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
\
train/gradients/b_count_3Exittrain/gradients/Switch_1*
_output_shapes
: *
T0
|
9train/gradients/loss/average_loss/average_loss_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
~
;train/gradients/loss/average_loss/average_loss_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Itrain/gradients/loss/average_loss/average_loss_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/loss/average_loss/average_loss_grad/Shape;train/gradients/loss/average_loss/average_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;train/gradients/loss/average_loss/average_loss_grad/RealDivRealDivtrain/gradients/Fill loss/average_loss/average_loss/y*
T0*
_output_shapes
: 
�
7train/gradients/loss/average_loss/average_loss_grad/SumSum;train/gradients/loss/average_loss/average_loss_grad/RealDivItrain/gradients/loss/average_loss/average_loss_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;train/gradients/loss/average_loss/average_loss_grad/ReshapeReshape7train/gradients/loss/average_loss/average_loss_grad/Sum9train/gradients/loss/average_loss/average_loss_grad/Shape*
_output_shapes
: *
T0*
Tshape0
{
7train/gradients/loss/average_loss/average_loss_grad/NegNegloss/average_loss/sum_loss*
T0*
_output_shapes
: 
�
=train/gradients/loss/average_loss/average_loss_grad/RealDiv_1RealDiv7train/gradients/loss/average_loss/average_loss_grad/Neg loss/average_loss/average_loss/y*
T0*
_output_shapes
: 
�
=train/gradients/loss/average_loss/average_loss_grad/RealDiv_2RealDiv=train/gradients/loss/average_loss/average_loss_grad/RealDiv_1 loss/average_loss/average_loss/y*
T0*
_output_shapes
: 
�
7train/gradients/loss/average_loss/average_loss_grad/mulMultrain/gradients/Fill=train/gradients/loss/average_loss/average_loss_grad/RealDiv_2*
T0*
_output_shapes
: 
�
9train/gradients/loss/average_loss/average_loss_grad/Sum_1Sum7train/gradients/loss/average_loss/average_loss_grad/mulKtrain/gradients/loss/average_loss/average_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=train/gradients/loss/average_loss/average_loss_grad/Reshape_1Reshape9train/gradients/loss/average_loss/average_loss_grad/Sum_1;train/gradients/loss/average_loss/average_loss_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Dtrain/gradients/loss/average_loss/average_loss_grad/tuple/group_depsNoOp<^train/gradients/loss/average_loss/average_loss_grad/Reshape>^train/gradients/loss/average_loss/average_loss_grad/Reshape_1
�
Ltrain/gradients/loss/average_loss/average_loss_grad/tuple/control_dependencyIdentity;train/gradients/loss/average_loss/average_loss_grad/ReshapeE^train/gradients/loss/average_loss/average_loss_grad/tuple/group_deps*
_output_shapes
: *
T0*N
_classD
B@loc:@train/gradients/loss/average_loss/average_loss_grad/Reshape
�
Ntrain/gradients/loss/average_loss/average_loss_grad/tuple/control_dependency_1Identity=train/gradients/loss/average_loss/average_loss_grad/Reshape_1E^train/gradients/loss/average_loss/average_loss_grad/tuple/group_deps*
_output_shapes
: *
T0*P
_classF
DBloc:@train/gradients/loss/average_loss/average_loss_grad/Reshape_1
�
=train/gradients/loss/average_loss/sum_loss_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
7train/gradients/loss/average_loss/sum_loss_grad/ReshapeReshapeLtrain/gradients/loss/average_loss/average_loss_grad/tuple/control_dependency=train/gradients/loss/average_loss/sum_loss_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
5train/gradients/loss/average_loss/sum_loss_grad/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
�
4train/gradients/loss/average_loss/sum_loss_grad/TileTile7train/gradients/loss/average_loss/sum_loss_grad/Reshape5train/gradients/loss/average_loss/sum_loss_grad/Const*
_output_shapes	
:�*

Tmultiples0*
T0
y
.train/gradients/loss/losses/truediv_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:�
{
0train/gradients/loss/losses/truediv_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
>train/gradients/loss/losses/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs.train/gradients/loss/losses/truediv_grad/Shape0train/gradients/loss/losses/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0train/gradients/loss/losses/truediv_grad/RealDivRealDiv4train/gradients/loss/average_loss/sum_loss_grad/Tileloss/losses/add*
T0*
_output_shapes	
:�
�
,train/gradients/loss/losses/truediv_grad/SumSum0train/gradients/loss/losses/truediv_grad/RealDiv>train/gradients/loss/losses/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
0train/gradients/loss/losses/truediv_grad/ReshapeReshape,train/gradients/loss/losses/truediv_grad/Sum.train/gradients/loss/losses/truediv_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
j
,train/gradients/loss/losses/truediv_grad/NegNegloss/losses/mul*
_output_shapes	
:�*
T0
�
2train/gradients/loss/losses/truediv_grad/RealDiv_1RealDiv,train/gradients/loss/losses/truediv_grad/Negloss/losses/add*
_output_shapes	
:�*
T0
�
2train/gradients/loss/losses/truediv_grad/RealDiv_2RealDiv2train/gradients/loss/losses/truediv_grad/RealDiv_1loss/losses/add*
T0*
_output_shapes	
:�
�
,train/gradients/loss/losses/truediv_grad/mulMul4train/gradients/loss/average_loss/sum_loss_grad/Tile2train/gradients/loss/losses/truediv_grad/RealDiv_2*
T0*
_output_shapes	
:�
�
.train/gradients/loss/losses/truediv_grad/Sum_1Sum,train/gradients/loss/losses/truediv_grad/mul@train/gradients/loss/losses/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
2train/gradients/loss/losses/truediv_grad/Reshape_1Reshape.train/gradients/loss/losses/truediv_grad/Sum_10train/gradients/loss/losses/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
9train/gradients/loss/losses/truediv_grad/tuple/group_depsNoOp1^train/gradients/loss/losses/truediv_grad/Reshape3^train/gradients/loss/losses/truediv_grad/Reshape_1
�
Atrain/gradients/loss/losses/truediv_grad/tuple/control_dependencyIdentity0train/gradients/loss/losses/truediv_grad/Reshape:^train/gradients/loss/losses/truediv_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/loss/losses/truediv_grad/Reshape*
_output_shapes	
:�
�
Ctrain/gradients/loss/losses/truediv_grad/tuple/control_dependency_1Identity2train/gradients/loss/losses/truediv_grad/Reshape_1:^train/gradients/loss/losses/truediv_grad/tuple/group_deps*
T0*E
_class;
97loc:@train/gradients/loss/losses/truediv_grad/Reshape_1*
_output_shapes	
:�
�
(train/gradients/loss/losses/mul_grad/MulMulAtrain/gradients/loss/losses/truediv_grad/tuple/control_dependency	loss/ones*
T0*
_output_shapes	
:�
�
*train/gradients/loss/losses/mul_grad/Mul_1MulAtrain/gradients/loss/losses/truediv_grad/tuple/control_dependencyloss/losses/Square*
_output_shapes	
:�*
T0
�
5train/gradients/loss/losses/mul_grad/tuple/group_depsNoOp)^train/gradients/loss/losses/mul_grad/Mul+^train/gradients/loss/losses/mul_grad/Mul_1
�
=train/gradients/loss/losses/mul_grad/tuple/control_dependencyIdentity(train/gradients/loss/losses/mul_grad/Mul6^train/gradients/loss/losses/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/loss/losses/mul_grad/Mul*
_output_shapes	
:�
�
?train/gradients/loss/losses/mul_grad/tuple/control_dependency_1Identity*train/gradients/loss/losses/mul_grad/Mul_16^train/gradients/loss/losses/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/loss/losses/mul_grad/Mul_1*
_output_shapes	
:�
�
-train/gradients/loss/losses/Square_grad/ConstConst>^train/gradients/loss/losses/mul_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
+train/gradients/loss/losses/Square_grad/MulMulloss/losses/Sub-train/gradients/loss/losses/Square_grad/Const*
T0*
_output_shapes	
:�
�
-train/gradients/loss/losses/Square_grad/Mul_1Mul=train/gradients/loss/losses/mul_grad/tuple/control_dependency+train/gradients/loss/losses/Square_grad/Mul*
T0*
_output_shapes	
:�
}
*train/gradients/loss/losses/Sub_grad/ShapeShapeloss/reshape_target*
T0*
out_type0*
_output_shapes
:
w
,train/gradients/loss/losses/Sub_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
:train/gradients/loss/losses/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/loss/losses/Sub_grad/Shape,train/gradients/loss/losses/Sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(train/gradients/loss/losses/Sub_grad/SumSum-train/gradients/loss/losses/Square_grad/Mul_1:train/gradients/loss/losses/Sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
,train/gradients/loss/losses/Sub_grad/ReshapeReshape(train/gradients/loss/losses/Sub_grad/Sum*train/gradients/loss/losses/Sub_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
�
*train/gradients/loss/losses/Sub_grad/Sum_1Sum-train/gradients/loss/losses/Square_grad/Mul_1<train/gradients/loss/losses/Sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
~
(train/gradients/loss/losses/Sub_grad/NegNeg*train/gradients/loss/losses/Sub_grad/Sum_1*
T0*
_output_shapes
:
�
.train/gradients/loss/losses/Sub_grad/Reshape_1Reshape(train/gradients/loss/losses/Sub_grad/Neg,train/gradients/loss/losses/Sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
5train/gradients/loss/losses/Sub_grad/tuple/group_depsNoOp-^train/gradients/loss/losses/Sub_grad/Reshape/^train/gradients/loss/losses/Sub_grad/Reshape_1
�
=train/gradients/loss/losses/Sub_grad/tuple/control_dependencyIdentity,train/gradients/loss/losses/Sub_grad/Reshape6^train/gradients/loss/losses/Sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/loss/losses/Sub_grad/Reshape*#
_output_shapes
:���������
�
?train/gradients/loss/losses/Sub_grad/tuple/control_dependency_1Identity.train/gradients/loss/losses/Sub_grad/Reshape_16^train/gradients/loss/losses/Sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/loss/losses/Sub_grad/Reshape_1*
_output_shapes	
:�
�
2train/gradients/loss/reshape_prediction_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�     
�
4train/gradients/loss/reshape_prediction_grad/ReshapeReshape?train/gradients/loss/losses/Sub_grad/tuple/control_dependency_12train/gradients/loss/reshape_prediction_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
3train/gradients/hidden_output/ys_out/Add_grad/ShapeConst*
valueB"�     *
dtype0*
_output_shapes
:

5train/gradients/hidden_output/ys_out/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
Ctrain/gradients/hidden_output/ys_out/Add_grad/BroadcastGradientArgsBroadcastGradientArgs3train/gradients/hidden_output/ys_out/Add_grad/Shape5train/gradients/hidden_output/ys_out/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1train/gradients/hidden_output/ys_out/Add_grad/SumSum4train/gradients/loss/reshape_prediction_grad/ReshapeCtrain/gradients/hidden_output/ys_out/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/hidden_output/ys_out/Add_grad/ReshapeReshape1train/gradients/hidden_output/ys_out/Add_grad/Sum3train/gradients/hidden_output/ys_out/Add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
3train/gradients/hidden_output/ys_out/Add_grad/Sum_1Sum4train/gradients/loss/reshape_prediction_grad/ReshapeEtrain/gradients/hidden_output/ys_out/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7train/gradients/hidden_output/ys_out/Add_grad/Reshape_1Reshape3train/gradients/hidden_output/ys_out/Add_grad/Sum_15train/gradients/hidden_output/ys_out/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
>train/gradients/hidden_output/ys_out/Add_grad/tuple/group_depsNoOp6^train/gradients/hidden_output/ys_out/Add_grad/Reshape8^train/gradients/hidden_output/ys_out/Add_grad/Reshape_1
�
Ftrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependencyIdentity5train/gradients/hidden_output/ys_out/Add_grad/Reshape?^train/gradients/hidden_output/ys_out/Add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/hidden_output/ys_out/Add_grad/Reshape*
_output_shapes
:	�
�
Htrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependency_1Identity7train/gradients/hidden_output/ys_out/Add_grad/Reshape_1?^train/gradients/hidden_output/ys_out/Add_grad/tuple/group_deps*
T0*J
_class@
><loc:@train/gradients/hidden_output/ys_out/Add_grad/Reshape_1*
_output_shapes
:
�
7train/gradients/hidden_output/ys_out/MatMul_grad/MatMulMatMulFtrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependencyhidden_output/Weights/read*
T0*
_output_shapes
:	�
*
transpose_a( *
transpose_b(
�
9train/gradients/hidden_output/ys_out/MatMul_grad/MatMul_1MatMulhidden_output/2_2DFtrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
Atrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/group_depsNoOp8^train/gradients/hidden_output/ys_out/MatMul_grad/MatMul:^train/gradients/hidden_output/ys_out/MatMul_grad/MatMul_1
�
Itrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/control_dependencyIdentity7train/gradients/hidden_output/ys_out/MatMul_grad/MatMulB^train/gradients/hidden_output/ys_out/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@train/gradients/hidden_output/ys_out/MatMul_grad/MatMul*
_output_shapes
:	�

�
Ktrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/control_dependency_1Identity9train/gradients/hidden_output/ys_out/MatMul_grad/MatMul_1B^train/gradients/hidden_output/ys_out/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/hidden_output/ys_out/MatMul_grad/MatMul_1*
_output_shapes

:

�
-train/gradients/hidden_output/2_2D_grad/ShapeConst*!
valueB"2      
   *
dtype0*
_output_shapes
:
�
/train/gradients/hidden_output/2_2D_grad/ReshapeReshapeItrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/control_dependency-train/gradients/hidden_output/2_2D_grad/Shape*
T0*
Tshape0*"
_output_shapes
:2

�
;train/gradients/cell/rnn/transpose_1_grad/InvertPermutationInvertPermutationcell/rnn/concat_2*
T0*
_output_shapes
:
�
3train/gradients/cell/rnn/transpose_1_grad/transpose	Transpose/train/gradients/hidden_output/2_2D_grad/Reshape;train/gradients/cell/rnn/transpose_1_grad/InvertPermutation*
T0*"
_output_shapes
:2
*
Tperm0
�
dtrain/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3cell/rnn/TensorArraycell/rnn/while/Exit_2*'
_class
loc:@cell/rnn/TensorArray*
sourcetrain/gradients*
_output_shapes

:: 
�
`train/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitycell/rnn/while/Exit_2e^train/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*'
_class
loc:@cell/rnn/TensorArray*
_output_shapes
: 
�
jtrain/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3dtrain/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3cell/rnn/TensorArrayStack/range3train/gradients/cell/rnn/transpose_1_grad/transpose`train/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
j
train/gradients/zerosConst*
valueB2
*    *
dtype0*
_output_shapes

:2

l
train/gradients/zeros_1Const*
valueB2
*    *
dtype0*
_output_shapes

:2

�
1train/gradients/cell/rnn/while/Exit_2_grad/b_exitEnterjtrain/gradients/cell/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context*
T0*
is_constant( 
�
1train/gradients/cell/rnn/while/Exit_3_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:2
*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
1train/gradients/cell/rnn/while/Exit_4_grad/b_exitEntertrain/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:2
*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
5train/gradients/cell/rnn/while/Switch_2_grad/b_switchMerge1train/gradients/cell/rnn/while/Exit_2_grad/b_exit<train/gradients/cell/rnn/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
�
5train/gradients/cell/rnn/while/Switch_3_grad/b_switchMerge1train/gradients/cell/rnn/while/Exit_3_grad/b_exit<train/gradients/cell/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N* 
_output_shapes
:2
: 
�
5train/gradients/cell/rnn/while/Switch_4_grad/b_switchMerge1train/gradients/cell/rnn/while/Exit_4_grad/b_exit<train/gradients/cell/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N* 
_output_shapes
:2
: 
�
2train/gradients/cell/rnn/while/Merge_2_grad/SwitchSwitch5train/gradients/cell/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 
y
<train/gradients/cell/rnn/while/Merge_2_grad/tuple/group_depsNoOp3^train/gradients/cell/rnn/while/Merge_2_grad/Switch
�
Dtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity2train/gradients/cell/rnn/while/Merge_2_grad/Switch=^train/gradients/cell/rnn/while/Merge_2_grad/tuple/group_deps*
_output_shapes
: *
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_2_grad/b_switch
�
Ftrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity4train/gradients/cell/rnn/while/Merge_2_grad/Switch:1=^train/gradients/cell/rnn/while/Merge_2_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
2train/gradients/cell/rnn/while/Merge_3_grad/SwitchSwitch5train/gradients/cell/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:2
:2

y
<train/gradients/cell/rnn/while/Merge_3_grad/tuple/group_depsNoOp3^train/gradients/cell/rnn/while/Merge_3_grad/Switch
�
Dtrain/gradients/cell/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity2train/gradients/cell/rnn/while/Merge_3_grad/Switch=^train/gradients/cell/rnn/while/Merge_3_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:2

�
Ftrain/gradients/cell/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity4train/gradients/cell/rnn/while/Merge_3_grad/Switch:1=^train/gradients/cell/rnn/while/Merge_3_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:2

�
2train/gradients/cell/rnn/while/Merge_4_grad/SwitchSwitch5train/gradients/cell/rnn/while/Switch_4_grad/b_switchtrain/gradients/b_count_2*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:2
:2

y
<train/gradients/cell/rnn/while/Merge_4_grad/tuple/group_depsNoOp3^train/gradients/cell/rnn/while/Merge_4_grad/Switch
�
Dtrain/gradients/cell/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity2train/gradients/cell/rnn/while/Merge_4_grad/Switch=^train/gradients/cell/rnn/while/Merge_4_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:2

�
Ftrain/gradients/cell/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity4train/gradients/cell/rnn/while/Merge_4_grad/Switch:1=^train/gradients/cell/rnn/while/Merge_4_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:2

�
0train/gradients/cell/rnn/while/Enter_2_grad/ExitExitDtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
0train/gradients/cell/rnn/while/Enter_3_grad/ExitExitDtrain/gradients/cell/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes

:2

�
0train/gradients/cell/rnn/while/Enter_4_grad/ExitExitDtrain/gradients/cell/rnn/while/Merge_4_grad/tuple/control_dependency*
_output_shapes

:2
*
T0
�
itrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3otrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterFtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2*
sourcetrain/gradients*
_output_shapes

:: 
�
otrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntercell/rnn/TensorArray*
T0*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
etrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityFtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1j^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*7
_class-
+)loc:@cell/rnn/while/basic_lstm_cell/Mul_2
�
Ytrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3itrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3dtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2etrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_output_shapes
:���������

�
_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*,
_class"
 loc:@cell/rnn/while/Identity_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*,
_class"
 loc:@cell/rnn/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
�
_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context*
T0*
is_constant(
�
etrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Entercell/rnn/while/Identity_1^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
dtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2jtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
�
jtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter_train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
`train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggere^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2O^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2I^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2K^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2I^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2K^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2G^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2I^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2M^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2O^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
Xtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpG^train/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1Z^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
`train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityYtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Y^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*l
_classb
`^loc:@train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes

:2

�
btrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityFtrain/gradients/cell/rnn/while/Merge_2_grad/tuple/control_dependency_1Y^train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
train/gradients/AddNAddNFtrain/gradients/cell/rnn/while/Merge_4_grad/tuple/control_dependency_1`train/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_4_grad/b_switch*
N*
_output_shapes

:2

�
=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/MulMultrain/gradients/AddNHtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes

:2

�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *;
_class1
/-loc:@cell/rnn/while/basic_lstm_cell/Sigmoid_2*
valueB :
���������
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*;
_class1
/-loc:@cell/rnn/while/basic_lstm_cell/Sigmoid_2
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter(cell/rnn/while/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
?train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1Multrain/gradients/AddNJtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
_output_shapes

:2
*
T0
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*8
_class.
,*loc:@cell/rnn/while/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*8
_class.
,*loc:@cell/rnn/while/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterEtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter%cell/rnn/while/basic_lstm_cell/Tanh_1^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:2

�
Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterEtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context*
T0*
is_constant(
�
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp>^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul@^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/MulK^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
_output_shapes

:2
*
T0*P
_classF
DBloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity?train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1K^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1*
_output_shapes

:2

�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradJtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
_output_shapes

:2
*
T0
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradHtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2

�
<train/gradients/cell/rnn/while/Switch_2_grad_1/NextIterationNextIterationbtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
train/gradients/AddN_1AddNFtrain/gradients/cell/rnn/while/Merge_3_grad/tuple/control_dependency_1Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes

:2

k
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^train/gradients/AddN_1
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentitytrain/gradients/AddN_1K^train/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:2

�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identitytrain/gradients/AddN_1K^train/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
_output_shapes

:2
*
T0*H
_class>
<:loc:@train/gradients/cell/rnn/while/Switch_3_grad/b_switch
�
;train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/MulMulRtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyFtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes

:2

�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*9
_class/
-+loc:@cell/rnn/while/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/Const*
	elem_type0*9
_class/
-+loc:@cell/rnn/while/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/EnterEnterAtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Atrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter&cell/rnn/while/basic_lstm_cell/Sigmoid^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Ftrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Ltrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:2

�
Ltrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterAtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1MulRtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyHtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:2

�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*,
_class"
 loc:@cell/rnn/while/Identity_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Const*
	elem_type0*,
_class"
 loc:@cell/rnn/while/Identity_3*

stack_name *
_output_shapes
:
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context*
T0*
is_constant(
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Entercell/rnn/while/Identity_3^train/gradients/Add*
_output_shapes

:2
*
swap_memory( *
T0
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp<^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul>^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1
�
Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity;train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/MulI^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul*
_output_shapes

:2

�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1I^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes

:2

�
=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/MulMulTtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes

:2

�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*6
_class,
*(loc:@cell/rnn/while/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Const*6
_class,
*(loc:@cell/rnn/while/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context*
T0*
is_constant(
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter#cell/rnn/while/basic_lstm_cell/Tanh^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:2

�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnterCtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context*
T0*
is_constant(
�
?train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1MulTtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:2

�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *;
_class1
/-loc:@cell/rnn/while/basic_lstm_cell/Sigmoid_1*
valueB :
���������
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*;
_class1
/-loc:@cell/rnn/while/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterEtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context*
T0*
is_constant(
�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter(cell/rnn/while/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterEtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Jtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp>^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul@^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity=train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/MulK^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul*
_output_shapes

:2

�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity?train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1K^train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1*
_output_shapes

:2

�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradFtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2

�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradJtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:2

�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradHtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2

�
<train/gradients/cell/rnn/while/Switch_3_grad_1/NextIterationNextIterationPtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0*
_output_shapes

:2

�
=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/ShapeConst^train/gradients/Sub*
valueB"2   
   *
dtype0*
_output_shapes
:
�
?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
�
Mtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgs=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/SumSumGtrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradMtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/ReshapeReshape;train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Sum=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape*
_output_shapes

:2
*
T0*
Tshape0
�
=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Sum_1SumGtrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradOtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape_1Reshape=train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Sum_1?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOp@^train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/ReshapeB^train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
Ptrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity?train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/ReshapeI^train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
_output_shapes

:2
*
T0*R
_classH
FDloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape
�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1IdentityAtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape_1I^train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@train/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 
�
@train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concatConcatV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradAtrain/gradients/cell/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradPtrain/gradients/cell/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyItrain/gradients/cell/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradFtrain/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes

:2(*

Tidx0
�
Ftrain/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad@train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:(
�
Ltrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpH^train/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradA^train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity@train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concatM^train/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes

:2(*
T0*S
_classI
GEloc:@train/gradients/cell/rnn/while/basic_lstm_cell/split_grad/concat
�
Vtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityGtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradM^train/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@train/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:(
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMulMatMulTtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyGtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
_output_shapes

:2*
transpose_a( *
transpose_b(*
T0
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter$cell/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulNtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:(*
transpose_a(*
transpose_b( 
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*8
_class.
,*loc:@cell/rnn/while/basic_lstm_cell/concat*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*8
_class.
,*loc:@cell/rnn/while/basic_lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterItrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Otrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter%cell/rnn/while/basic_lstm_cell/concat^train/gradients/Add*
T0*
_output_shapes

:2*
swap_memory( 
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:2*
	elem_type0
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterItrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpB^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMulD^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
�
Strain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityAtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMulL^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
_output_shapes

:2*
T0*T
_classJ
HFloc:@train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul
�
Utrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityCtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1L^train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@train/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:(
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes
:(*
valueB(*    
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterGtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:(*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeItrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Otrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:(: 
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchItrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*
T0* 
_output_shapes
:(:(
�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddJtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1Vtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:(
�
Otrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationEtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:(
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitHtrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:(
�
@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
?train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
>train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/modFloorMod@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Const?train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeShape cell/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeNShapeNLtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Ntrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/ConstConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@cell/rnn/while/TensorArrayReadV3*
valueB :
���������
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const*

stack_name *
_output_shapes
:*
	elem_type0*3
_class)
'%loc:@cell/rnn/while/TensorArrayReadV3
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/EnterEnterGtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Mtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter cell/rnn/while/TensorArrayReadV3^train/gradients/Add*
T0*'
_output_shapes
:���������
*
swap_memory( 
�
Ltrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Rtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*'
_output_shapes
:���������

�
Rtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterGtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1Const*,
_class"
 loc:@cell/rnn/while/Identity_4*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1*,
_class"
 loc:@cell/rnn/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0
�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterItrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecell/rnn/while/while_context
�
Otrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1cell/rnn/while/Identity_4^train/gradients/Add*
T0*
_output_shapes

:2
*
swap_memory( 
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2Ttrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^train/gradients/Sub*
_output_shapes

:2
*
	elem_type0
�
Ttrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterItrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffset>train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/modAtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeNCtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/SliceSliceStrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyGtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetAtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:������������������
�
Btrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice_1SliceStrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyItrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset:1Ctrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*0
_output_shapes
:������������������
�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOpA^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/SliceC^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice_1
�
Strain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentity@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/SliceL^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*S
_classI
GEloc:@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice
�
Utrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityBtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice_1L^train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:2

�
Ftrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes

:(
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterFtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:(*<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeHtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Ntrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:(: 
�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchHtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
Dtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/AddAddItrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1Utrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:(
�
Ntrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationDtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:(
�
Htrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitGtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
Wtrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3]train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^train/gradients/Sub*
_output_shapes

:: *9
_class/
-+loc:@cell/rnn/while/TensorArrayReadV3/Enter*
sourcetrain/gradients
�
]train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntercell/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*<

frame_name.,train/gradients/cell/rnn/while/while_context*
T0*9
_class/
-+loc:@cell/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
_train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterCcell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*9
_class/
-+loc:@cell/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Strain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity_train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1X^train/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*9
_class/
-+loc:@cell/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
�
Ytrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Wtrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3dtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Strain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyStrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
�
Ctrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Etrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterCtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *<

frame_name.,train/gradients/cell/rnn/while/while_context
�
Etrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeEtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Ktrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Dtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchEtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2train/gradients/b_count_2*
T0*
_output_shapes
: : 
�
Atrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddFtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Ytrain/gradients/cell/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
Ktrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationAtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
�
Etrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitDtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
�
<train/gradients/cell/rnn/while/Switch_4_grad_1/NextIterationNextIterationUtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2

�
ztrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3cell/rnn/TensorArray_1Etrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*)
_class
loc:@cell/rnn/TensorArray_1*
sourcetrain/gradients*
_output_shapes

:: 
�
vtrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityEtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3{^train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@cell/rnn/TensorArray_1*
_output_shapes
: 
�
ltrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3ztrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3!cell/rnn/TensorArrayUnstack/rangevtrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*4
_output_shapes"
 :������������������
*
element_shape:
�
itrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpm^train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3F^train/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
qtrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityltrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3j^train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*+
_output_shapes
:���������
*
T0*
_classu
sqloc:@train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
strain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityEtrain/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3j^train/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*X
_classN
LJloc:@train/gradients/cell/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
�
9train/gradients/cell/rnn/transpose_grad/InvertPermutationInvertPermutationcell/rnn/concat*
_output_shapes
:*
T0
�
1train/gradients/cell/rnn/transpose_grad/transpose	Transposeqtrain/gradients/cell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency9train/gradients/cell/rnn/transpose_grad/InvertPermutation*+
_output_shapes
:���������
*
Tperm0*
T0
�
,train/gradients/hidden_input/2_3D_grad/ShapeShapehidden_input/ys_in/Add*
T0*
out_type0*
_output_shapes
:
�
.train/gradients/hidden_input/2_3D_grad/ReshapeReshape1train/gradients/cell/rnn/transpose_grad/transpose,train/gradients/hidden_input/2_3D_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
1train/gradients/hidden_input/ys_in/Add_grad/ShapeShapehidden_input/ys_in/MatMul*
T0*
out_type0*
_output_shapes
:
}
3train/gradients/hidden_input/ys_in/Add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
Atrain/gradients/hidden_input/ys_in/Add_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/hidden_input/ys_in/Add_grad/Shape3train/gradients/hidden_input/ys_in/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/train/gradients/hidden_input/ys_in/Add_grad/SumSum.train/gradients/hidden_input/2_3D_grad/ReshapeAtrain/gradients/hidden_input/ys_in/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
3train/gradients/hidden_input/ys_in/Add_grad/ReshapeReshape/train/gradients/hidden_input/ys_in/Add_grad/Sum1train/gradients/hidden_input/ys_in/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
1train/gradients/hidden_input/ys_in/Add_grad/Sum_1Sum.train/gradients/hidden_input/2_3D_grad/ReshapeCtrain/gradients/hidden_input/ys_in/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/hidden_input/ys_in/Add_grad/Reshape_1Reshape1train/gradients/hidden_input/ys_in/Add_grad/Sum_13train/gradients/hidden_input/ys_in/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

�
<train/gradients/hidden_input/ys_in/Add_grad/tuple/group_depsNoOp4^train/gradients/hidden_input/ys_in/Add_grad/Reshape6^train/gradients/hidden_input/ys_in/Add_grad/Reshape_1
�
Dtrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependencyIdentity3train/gradients/hidden_input/ys_in/Add_grad/Reshape=^train/gradients/hidden_input/ys_in/Add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/hidden_input/ys_in/Add_grad/Reshape*'
_output_shapes
:���������

�
Ftrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependency_1Identity5train/gradients/hidden_input/ys_in/Add_grad/Reshape_1=^train/gradients/hidden_input/ys_in/Add_grad/tuple/group_deps*
_output_shapes
:
*
T0*H
_class>
<:loc:@train/gradients/hidden_input/ys_in/Add_grad/Reshape_1
�
5train/gradients/hidden_input/ys_in/MatMul_grad/MatMulMatMulDtrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependencyhidden_input/Weights/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
7train/gradients/hidden_input/ys_in/MatMul_grad/MatMul_1MatMulhidden_input/2_2DDtrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
�
?train/gradients/hidden_input/ys_in/MatMul_grad/tuple/group_depsNoOp6^train/gradients/hidden_input/ys_in/MatMul_grad/MatMul8^train/gradients/hidden_input/ys_in/MatMul_grad/MatMul_1
�
Gtrain/gradients/hidden_input/ys_in/MatMul_grad/tuple/control_dependencyIdentity5train/gradients/hidden_input/ys_in/MatMul_grad/MatMul@^train/gradients/hidden_input/ys_in/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/hidden_input/ys_in/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Itrain/gradients/hidden_input/ys_in/MatMul_grad/tuple/control_dependency_1Identity7train/gradients/hidden_input/ys_in/MatMul_grad/MatMul_1@^train/gradients/hidden_input/ys_in/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@train/gradients/hidden_input/ys_in/MatMul_grad/MatMul_1*
_output_shapes

:

�
train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB
 *fff?
�
train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
	container *
shape: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias
�
train/beta1_power/readIdentitytrain/beta1_power*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
: 
�
train/beta2_power/readIdentitytrain/beta2_power*
_output_shapes
: *
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias
�
+hidden_input/Weights/Adam/Initializer/zerosConst*'
_class
loc:@hidden_input/Weights*
valueB
*    *
dtype0*
_output_shapes

:

�
hidden_input/Weights/Adam
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *'
_class
loc:@hidden_input/Weights*
	container *
shape
:

�
 hidden_input/Weights/Adam/AssignAssignhidden_input/Weights/Adam+hidden_input/Weights/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*'
_class
loc:@hidden_input/Weights
�
hidden_input/Weights/Adam/readIdentityhidden_input/Weights/Adam*
T0*'
_class
loc:@hidden_input/Weights*
_output_shapes

:

�
-hidden_input/Weights/Adam_1/Initializer/zerosConst*'
_class
loc:@hidden_input/Weights*
valueB
*    *
dtype0*
_output_shapes

:

�
hidden_input/Weights/Adam_1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *'
_class
loc:@hidden_input/Weights*
	container 
�
"hidden_input/Weights/Adam_1/AssignAssignhidden_input/Weights/Adam_1-hidden_input/Weights/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*'
_class
loc:@hidden_input/Weights
�
 hidden_input/Weights/Adam_1/readIdentityhidden_input/Weights/Adam_1*
T0*'
_class
loc:@hidden_input/Weights*
_output_shapes

:

�
*hidden_input/biases/Adam/Initializer/zerosConst*&
_class
loc:@hidden_input/biases*
valueB
*    *
dtype0*
_output_shapes
:

�
hidden_input/biases/Adam
VariableV2*
shared_name *&
_class
loc:@hidden_input/biases*
	container *
shape:
*
dtype0*
_output_shapes
:

�
hidden_input/biases/Adam/AssignAssignhidden_input/biases/Adam*hidden_input/biases/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@hidden_input/biases*
validate_shape(*
_output_shapes
:

�
hidden_input/biases/Adam/readIdentityhidden_input/biases/Adam*
T0*&
_class
loc:@hidden_input/biases*
_output_shapes
:

�
,hidden_input/biases/Adam_1/Initializer/zerosConst*&
_class
loc:@hidden_input/biases*
valueB
*    *
dtype0*
_output_shapes
:

�
hidden_input/biases/Adam_1
VariableV2*
shared_name *&
_class
loc:@hidden_input/biases*
	container *
shape:
*
dtype0*
_output_shapes
:

�
!hidden_input/biases/Adam_1/AssignAssignhidden_input/biases/Adam_1,hidden_input/biases/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@hidden_input/biases*
validate_shape(*
_output_shapes
:

�
hidden_input/biases/Adam_1/readIdentityhidden_input/biases/Adam_1*
T0*&
_class
loc:@hidden_input/biases*
_output_shapes
:

�
6cell/rnn/basic_lstm_cell/kernel/Adam/Initializer/zerosConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB(*    *
dtype0*
_output_shapes

:(
�
$cell/rnn/basic_lstm_cell/kernel/Adam
VariableV2*
shared_name *2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(
�
+cell/rnn/basic_lstm_cell/kernel/Adam/AssignAssign$cell/rnn/basic_lstm_cell/kernel/Adam6cell/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
)cell/rnn/basic_lstm_cell/kernel/Adam/readIdentity$cell/rnn/basic_lstm_cell/kernel/Adam*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
8cell/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
valueB(*    *
dtype0*
_output_shapes

:(
�
&cell/rnn/basic_lstm_cell/kernel/Adam_1
VariableV2*
shared_name *2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(
�
-cell/rnn/basic_lstm_cell/kernel/Adam_1/AssignAssign&cell/rnn/basic_lstm_cell/kernel/Adam_18cell/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
+cell/rnn/basic_lstm_cell/kernel/Adam_1/readIdentity&cell/rnn/basic_lstm_cell/kernel/Adam_1*
_output_shapes

:(*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel
�
4cell/rnn/basic_lstm_cell/bias/Adam/Initializer/zerosConst*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
"cell/rnn/basic_lstm_cell/bias/Adam
VariableV2*
shape:(*
dtype0*
_output_shapes
:(*
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
	container 
�
)cell/rnn/basic_lstm_cell/bias/Adam/AssignAssign"cell/rnn/basic_lstm_cell/bias/Adam4cell/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
'cell/rnn/basic_lstm_cell/bias/Adam/readIdentity"cell/rnn/basic_lstm_cell/bias/Adam*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
:(
�
6cell/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
$cell/rnn/basic_lstm_cell/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
	container *
shape:(
�
+cell/rnn/basic_lstm_cell/bias/Adam_1/AssignAssign$cell/rnn/basic_lstm_cell/bias/Adam_16cell/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
)cell/rnn/basic_lstm_cell/bias/Adam_1/readIdentity$cell/rnn/basic_lstm_cell/bias/Adam_1*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
:(
�
,hidden_output/Weights/Adam/Initializer/zerosConst*(
_class
loc:@hidden_output/Weights*
valueB
*    *
dtype0*
_output_shapes

:

�
hidden_output/Weights/Adam
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *(
_class
loc:@hidden_output/Weights*
	container *
shape
:

�
!hidden_output/Weights/Adam/AssignAssignhidden_output/Weights/Adam,hidden_output/Weights/Adam/Initializer/zeros*
T0*(
_class
loc:@hidden_output/Weights*
validate_shape(*
_output_shapes

:
*
use_locking(
�
hidden_output/Weights/Adam/readIdentityhidden_output/Weights/Adam*
_output_shapes

:
*
T0*(
_class
loc:@hidden_output/Weights
�
.hidden_output/Weights/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*(
_class
loc:@hidden_output/Weights*
valueB
*    
�
hidden_output/Weights/Adam_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *(
_class
loc:@hidden_output/Weights*
	container *
shape
:

�
#hidden_output/Weights/Adam_1/AssignAssignhidden_output/Weights/Adam_1.hidden_output/Weights/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*(
_class
loc:@hidden_output/Weights
�
!hidden_output/Weights/Adam_1/readIdentityhidden_output/Weights/Adam_1*
T0*(
_class
loc:@hidden_output/Weights*
_output_shapes

:

�
+hidden_output/biases/Adam/Initializer/zerosConst*'
_class
loc:@hidden_output/biases*
valueB*    *
dtype0*
_output_shapes
:
�
hidden_output/biases/Adam
VariableV2*'
_class
loc:@hidden_output/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
 hidden_output/biases/Adam/AssignAssignhidden_output/biases/Adam+hidden_output/biases/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@hidden_output/biases*
validate_shape(*
_output_shapes
:
�
hidden_output/biases/Adam/readIdentityhidden_output/biases/Adam*
_output_shapes
:*
T0*'
_class
loc:@hidden_output/biases
�
-hidden_output/biases/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*'
_class
loc:@hidden_output/biases*
valueB*    
�
hidden_output/biases/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@hidden_output/biases*
	container *
shape:
�
"hidden_output/biases/Adam_1/AssignAssignhidden_output/biases/Adam_1-hidden_output/biases/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@hidden_output/biases*
validate_shape(*
_output_shapes
:
�
 hidden_output/biases/Adam_1/readIdentityhidden_output/biases/Adam_1*
T0*'
_class
loc:@hidden_output/biases*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *���;*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
0train/Adam/update_hidden_input/Weights/ApplyAdam	ApplyAdamhidden_input/Weightshidden_input/Weights/Adamhidden_input/Weights/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonItrain/gradients/hidden_input/ys_in/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@hidden_input/Weights*
use_nesterov( *
_output_shapes

:

�
/train/Adam/update_hidden_input/biases/ApplyAdam	ApplyAdamhidden_input/biaseshidden_input/biases/Adamhidden_input/biases/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonFtrain/gradients/hidden_input/ys_in/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@hidden_input/biases*
use_nesterov( *
_output_shapes
:

�
;train/Adam/update_cell/rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamcell/rnn/basic_lstm_cell/kernel$cell/rnn/basic_lstm_cell/kernel/Adam&cell/rnn/basic_lstm_cell/kernel/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonHtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*2
_class(
&$loc:@cell/rnn/basic_lstm_cell/kernel*
use_nesterov( *
_output_shapes

:(*
use_locking( 
�
9train/Adam/update_cell/rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamcell/rnn/basic_lstm_cell/bias"cell/rnn/basic_lstm_cell/bias/Adam$cell/rnn/basic_lstm_cell/bias/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonItrain/gradients/cell/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes
:(
�
1train/Adam/update_hidden_output/Weights/ApplyAdam	ApplyAdamhidden_output/Weightshidden_output/Weights/Adamhidden_output/Weights/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonKtrain/gradients/hidden_output/ys_out/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@hidden_output/Weights*
use_nesterov( *
_output_shapes

:

�
0train/Adam/update_hidden_output/biases/ApplyAdam	ApplyAdamhidden_output/biaseshidden_output/biases/Adamhidden_output/biases/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonHtrain/gradients/hidden_output/ys_out/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@hidden_output/biases*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1:^train/Adam/update_cell/rnn/basic_lstm_cell/bias/ApplyAdam<^train/Adam/update_cell/rnn/basic_lstm_cell/kernel/ApplyAdam1^train/Adam/update_hidden_input/Weights/ApplyAdam0^train/Adam/update_hidden_input/biases/ApplyAdam2^train/Adam/update_hidden_output/Weights/ApplyAdam1^train/Adam/update_hidden_output/biases/ApplyAdam*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2:^train/Adam/update_cell/rnn/basic_lstm_cell/bias/ApplyAdam<^train/Adam/update_cell/rnn/basic_lstm_cell/kernel/ApplyAdam1^train/Adam/update_hidden_input/Weights/ApplyAdam0^train/Adam/update_hidden_input/biases/ApplyAdam2^train/Adam/update_hidden_output/Weights/ApplyAdam1^train/Adam/update_hidden_output/biases/ApplyAdam*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*0
_class&
$"loc:@cell/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1:^train/Adam/update_cell/rnn/basic_lstm_cell/bias/ApplyAdam<^train/Adam/update_cell/rnn/basic_lstm_cell/kernel/ApplyAdam1^train/Adam/update_hidden_input/Weights/ApplyAdam0^train/Adam/update_hidden_input/biases/ApplyAdam2^train/Adam/update_hidden_output/Weights/ApplyAdam1^train/Adam/update_hidden_output/biases/ApplyAdam
[
Merge/MergeSummaryMergeSummaryloss/average_loss/loss*
N*
_output_shapes
: ""
train_op


train/Adam"�>
while_context�>�>
�>
cell/rnn/while/while_context *cell/rnn/while/LoopCond:02cell/rnn/while/Merge:0:cell/rnn/while/Identity:0Bcell/rnn/while/Exit:0Bcell/rnn/while/Exit_1:0Bcell/rnn/while/Exit_2:0Bcell/rnn/while/Exit_3:0Bcell/rnn/while/Exit_4:0Btrain/gradients/f_count_2:0J�:
cell/rnn/Minimum:0
cell/rnn/TensorArray:0
Ecell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
cell/rnn/TensorArray_1:0
$cell/rnn/basic_lstm_cell/bias/read:0
&cell/rnn/basic_lstm_cell/kernel/read:0
cell/rnn/strided_slice_1:0
cell/rnn/while/Enter:0
cell/rnn/while/Enter_1:0
cell/rnn/while/Enter_2:0
cell/rnn/while/Enter_3:0
cell/rnn/while/Enter_4:0
cell/rnn/while/Exit:0
cell/rnn/while/Exit_1:0
cell/rnn/while/Exit_2:0
cell/rnn/while/Exit_3:0
cell/rnn/while/Exit_4:0
cell/rnn/while/Identity:0
cell/rnn/while/Identity_1:0
cell/rnn/while/Identity_2:0
cell/rnn/while/Identity_3:0
cell/rnn/while/Identity_4:0
cell/rnn/while/Less/Enter:0
cell/rnn/while/Less:0
cell/rnn/while/Less_1/Enter:0
cell/rnn/while/Less_1:0
cell/rnn/while/LogicalAnd:0
cell/rnn/while/LoopCond:0
cell/rnn/while/Merge:0
cell/rnn/while/Merge:1
cell/rnn/while/Merge_1:0
cell/rnn/while/Merge_1:1
cell/rnn/while/Merge_2:0
cell/rnn/while/Merge_2:1
cell/rnn/while/Merge_3:0
cell/rnn/while/Merge_3:1
cell/rnn/while/Merge_4:0
cell/rnn/while/Merge_4:1
cell/rnn/while/NextIteration:0
 cell/rnn/while/NextIteration_1:0
 cell/rnn/while/NextIteration_2:0
 cell/rnn/while/NextIteration_3:0
 cell/rnn/while/NextIteration_4:0
cell/rnn/while/Switch:0
cell/rnn/while/Switch:1
cell/rnn/while/Switch_1:0
cell/rnn/while/Switch_1:1
cell/rnn/while/Switch_2:0
cell/rnn/while/Switch_2:1
cell/rnn/while/Switch_3:0
cell/rnn/while/Switch_3:1
cell/rnn/while/Switch_4:0
cell/rnn/while/Switch_4:1
(cell/rnn/while/TensorArrayReadV3/Enter:0
*cell/rnn/while/TensorArrayReadV3/Enter_1:0
"cell/rnn/while/TensorArrayReadV3:0
:cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
4cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
cell/rnn/while/add/y:0
cell/rnn/while/add:0
cell/rnn/while/add_1/y:0
cell/rnn/while/add_1:0
$cell/rnn/while/basic_lstm_cell/Add:0
&cell/rnn/while/basic_lstm_cell/Add_1:0
.cell/rnn/while/basic_lstm_cell/BiasAdd/Enter:0
(cell/rnn/while/basic_lstm_cell/BiasAdd:0
&cell/rnn/while/basic_lstm_cell/Const:0
(cell/rnn/while/basic_lstm_cell/Const_1:0
(cell/rnn/while/basic_lstm_cell/Const_2:0
-cell/rnn/while/basic_lstm_cell/MatMul/Enter:0
'cell/rnn/while/basic_lstm_cell/MatMul:0
$cell/rnn/while/basic_lstm_cell/Mul:0
&cell/rnn/while/basic_lstm_cell/Mul_1:0
&cell/rnn/while/basic_lstm_cell/Mul_2:0
(cell/rnn/while/basic_lstm_cell/Sigmoid:0
*cell/rnn/while/basic_lstm_cell/Sigmoid_1:0
*cell/rnn/while/basic_lstm_cell/Sigmoid_2:0
%cell/rnn/while/basic_lstm_cell/Tanh:0
'cell/rnn/while/basic_lstm_cell/Tanh_1:0
,cell/rnn/while/basic_lstm_cell/concat/axis:0
'cell/rnn/while/basic_lstm_cell/concat:0
&cell/rnn/while/basic_lstm_cell/split:0
&cell/rnn/while/basic_lstm_cell/split:1
&cell/rnn/while/basic_lstm_cell/split:2
&cell/rnn/while/basic_lstm_cell/split:3
train/gradients/Add/y:0
train/gradients/Add:0
train/gradients/Merge:0
train/gradients/Merge:1
train/gradients/NextIteration:0
train/gradients/Switch:0
train/gradients/Switch:1
atrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
gtrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
atrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
Qtrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2:0
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0
Mtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2:0
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2:0
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0
Mtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2:0
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0
Itrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2:0
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2:0
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0
Btrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/Shape:0
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter:0
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0
Otrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2:0
Qtrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc:0
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0
train/gradients/f_count:0
train/gradients/f_count_1:0
train/gradients/f_count_2:0�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0V
$cell/rnn/basic_lstm_cell/bias/read:0.cell/rnn/while/basic_lstm_cell/BiasAdd/Enter:09
cell/rnn/strided_slice_1:0cell/rnn/while/Less/Enter:0�
Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0Gtrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0W
&cell/rnn/basic_lstm_cell/kernel/read:0-cell/rnn/while/basic_lstm_cell/MatMul/Enter:0�
Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0Ctrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0Ktrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0�
Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc:0Itrain/gradients/cell/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter:0D
cell/rnn/TensorArray_1:0(cell/rnn/while/TensorArrayReadV3/Enter:0�
atrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0atrain/gradients/cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0T
cell/rnn/TensorArray:0:cell/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0s
Ecell/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0*cell/rnn/while/TensorArrayReadV3/Enter_1:0�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:03
cell/rnn/Minimum:0cell/rnn/while/Less_1/Enter:0�
Ktrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0Ktrain/gradients/cell/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0�
Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0Etrain/gradients/cell/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0Rcell/rnn/while/Enter:0Rcell/rnn/while/Enter_1:0Rcell/rnn/while/Enter_2:0Rcell/rnn/while/Enter_3:0Rcell/rnn/while/Enter_4:0Rtrain/gradients/f_count_1:0Zcell/rnn/strided_slice_1:0"�
	variables��
�
hidden_input/Weights:0hidden_input/Weights/Assignhidden_input/Weights/read:020hidden_input/Weights/Initializer/random_normal:08
z
hidden_input/biases:0hidden_input/biases/Assignhidden_input/biases/read:02'hidden_input/biases/Initializer/Const:08
�
!cell/rnn/basic_lstm_cell/kernel:0&cell/rnn/basic_lstm_cell/kernel/Assign&cell/rnn/basic_lstm_cell/kernel/read:02<cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform:08
�
cell/rnn/basic_lstm_cell/bias:0$cell/rnn/basic_lstm_cell/bias/Assign$cell/rnn/basic_lstm_cell/bias/read:021cell/rnn/basic_lstm_cell/bias/Initializer/zeros:08
�
hidden_output/Weights:0hidden_output/Weights/Assignhidden_output/Weights/read:021hidden_output/Weights/Initializer/random_normal:08
~
hidden_output/biases:0hidden_output/biases/Assignhidden_output/biases/read:02(hidden_output/biases/Initializer/Const:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
hidden_input/Weights/Adam:0 hidden_input/Weights/Adam/Assign hidden_input/Weights/Adam/read:02-hidden_input/Weights/Adam/Initializer/zeros:0
�
hidden_input/Weights/Adam_1:0"hidden_input/Weights/Adam_1/Assign"hidden_input/Weights/Adam_1/read:02/hidden_input/Weights/Adam_1/Initializer/zeros:0
�
hidden_input/biases/Adam:0hidden_input/biases/Adam/Assignhidden_input/biases/Adam/read:02,hidden_input/biases/Adam/Initializer/zeros:0
�
hidden_input/biases/Adam_1:0!hidden_input/biases/Adam_1/Assign!hidden_input/biases/Adam_1/read:02.hidden_input/biases/Adam_1/Initializer/zeros:0
�
&cell/rnn/basic_lstm_cell/kernel/Adam:0+cell/rnn/basic_lstm_cell/kernel/Adam/Assign+cell/rnn/basic_lstm_cell/kernel/Adam/read:028cell/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros:0
�
(cell/rnn/basic_lstm_cell/kernel/Adam_1:0-cell/rnn/basic_lstm_cell/kernel/Adam_1/Assign-cell/rnn/basic_lstm_cell/kernel/Adam_1/read:02:cell/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
�
$cell/rnn/basic_lstm_cell/bias/Adam:0)cell/rnn/basic_lstm_cell/bias/Adam/Assign)cell/rnn/basic_lstm_cell/bias/Adam/read:026cell/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros:0
�
&cell/rnn/basic_lstm_cell/bias/Adam_1:0+cell/rnn/basic_lstm_cell/bias/Adam_1/Assign+cell/rnn/basic_lstm_cell/bias/Adam_1/read:028cell/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
�
hidden_output/Weights/Adam:0!hidden_output/Weights/Adam/Assign!hidden_output/Weights/Adam/read:02.hidden_output/Weights/Adam/Initializer/zeros:0
�
hidden_output/Weights/Adam_1:0#hidden_output/Weights/Adam_1/Assign#hidden_output/Weights/Adam_1/read:020hidden_output/Weights/Adam_1/Initializer/zeros:0
�
hidden_output/biases/Adam:0 hidden_output/biases/Adam/Assign hidden_output/biases/Adam/read:02-hidden_output/biases/Adam/Initializer/zeros:0
�
hidden_output/biases/Adam_1:0"hidden_output/biases/Adam_1/Assign"hidden_output/biases/Adam_1/read:02/hidden_output/biases/Adam_1/Initializer/zeros:0")
	summaries

loss/average_loss/loss:0"�
trainable_variables��
�
hidden_input/Weights:0hidden_input/Weights/Assignhidden_input/Weights/read:020hidden_input/Weights/Initializer/random_normal:08
z
hidden_input/biases:0hidden_input/biases/Assignhidden_input/biases/read:02'hidden_input/biases/Initializer/Const:08
�
!cell/rnn/basic_lstm_cell/kernel:0&cell/rnn/basic_lstm_cell/kernel/Assign&cell/rnn/basic_lstm_cell/kernel/read:02<cell/rnn/basic_lstm_cell/kernel/Initializer/random_uniform:08
�
cell/rnn/basic_lstm_cell/bias:0$cell/rnn/basic_lstm_cell/bias/Assign$cell/rnn/basic_lstm_cell/bias/read:021cell/rnn/basic_lstm_cell/bias/Initializer/zeros:08
�
hidden_output/Weights:0hidden_output/Weights/Assignhidden_output/Weights/read:021hidden_output/Weights/Initializer/random_normal:08
~
hidden_output/biases:0hidden_output/biases/Assignhidden_output/biases/read:02(hidden_output/biases/Initializer/Const:08y�1�*       ����	zj	R1(�A*

loss/average_loss/lossd�/AU���,       ���E	��T1(�A*

loss/average_loss/lossOՉ@�:�,       ���E	�\%V1(�A(*

loss/average_loss/loss,2�@�
�,       ���E	��EX1(�A<*

loss/average_loss/loss�W'@�4,       ���E	��[Z1(�AP*

loss/average_loss/loss���?$��,       ���E	�p\1(�Ad*

loss/average_loss/loss;r?��M�,       ���E	y9�^1(�Ax*

loss/average_loss/lossɢ�>ݑ��-       <A��	$��`1(�A�*

loss/average_loss/loss���?���-       <A��	�b1(�A�*

loss/average_loss/lossSJ�>���-       <A��	�~e1(�A�*

loss/average_loss/loss���><�H