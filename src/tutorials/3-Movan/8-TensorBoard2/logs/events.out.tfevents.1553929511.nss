       �K"	  �I�'�Abrain.Event:2��1�L\      6�&	�A�I�'�A"��
q
inputs/x_inputPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
q
inputs/y_inputPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
s
"layer1/weights/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
f
!layer1/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
#layer1/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
1layer1/weights/random_normal/RandomStandardNormalRandomStandardNormal"layer1/weights/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
 layer1/weights/random_normal/mulMul1layer1/weights/random_normal/RandomStandardNormal#layer1/weights/random_normal/stddev*
T0*
_output_shapes

:

�
layer1/weights/random_normalAdd layer1/weights/random_normal/mul!layer1/weights/random_normal/mean*
T0*
_output_shapes

:

�
layer1/weights/W
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
layer1/weights/W/AssignAssignlayer1/weights/Wlayer1/weights/random_normal*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@layer1/weights/W
�
layer1/weights/W/readIdentitylayer1/weights/W*
T0*#
_class
loc:@layer1/weights/W*
_output_shapes

:


!layer1/weights/layer1/weights/tagConst*.
value%B# Blayer1/weights/layer1/weights*
dtype0*
_output_shapes
: 
�
layer1/weights/layer1/weightsHistogramSummary!layer1/weights/layer1/weights/taglayer1/weights/W/read*
T0*
_output_shapes
: 
h
layer1/biases/zerosConst*
valueB
*    *
dtype0*
_output_shapes

:

X
layer1/biases/add/yConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
k
layer1/biases/addAddlayer1/biases/zeroslayer1/biases/add/y*
T0*
_output_shapes

:

�
layer1/biases/b
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
layer1/biases/b/AssignAssignlayer1/biases/blayer1/biases/add*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*"
_class
loc:@layer1/biases/b
~
layer1/biases/b/readIdentitylayer1/biases/b*
T0*"
_class
loc:@layer1/biases/b*
_output_shapes

:

{
layer1/biases/layer1/biases/tagConst*,
value#B! Blayer1/biases/layer1/biases*
dtype0*
_output_shapes
: 
�
layer1/biases/layer1/biasesHistogramSummarylayer1/biases/layer1/biases/taglayer1/biases/b/read*
T0*
_output_shapes
: 
�
layer1/y/MatMulMatMulinputs/x_inputlayer1/weights/W/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
l
layer1/y/AddAddlayer1/y/MatMullayer1/biases/b/read*
T0*'
_output_shapes
:���������

S
layer1/ReluRelulayer1/y/Add*'
_output_shapes
:���������
*
T0
o
layer1/layer1/outputs/tagConst*
dtype0*
_output_shapes
: *&
valueB Blayer1/layer1/outputs
r
layer1/layer1/outputsHistogramSummarylayer1/layer1/outputs/taglayer1/Relu*
_output_shapes
: *
T0
s
"layer2/weights/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"
      
f
!layer2/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
#layer2/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
1layer2/weights/random_normal/RandomStandardNormalRandomStandardNormal"layer2/weights/random_normal/shape*

seed *
T0*
dtype0*
_output_shapes

:
*
seed2 
�
 layer2/weights/random_normal/mulMul1layer2/weights/random_normal/RandomStandardNormal#layer2/weights/random_normal/stddev*
T0*
_output_shapes

:

�
layer2/weights/random_normalAdd layer2/weights/random_normal/mul!layer2/weights/random_normal/mean*
_output_shapes

:
*
T0
�
layer2/weights/W
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
layer2/weights/W/AssignAssignlayer2/weights/Wlayer2/weights/random_normal*
T0*#
_class
loc:@layer2/weights/W*
validate_shape(*
_output_shapes

:
*
use_locking(
�
layer2/weights/W/readIdentitylayer2/weights/W*
T0*#
_class
loc:@layer2/weights/W*
_output_shapes

:


!layer2/weights/layer2/weights/tagConst*.
value%B# Blayer2/weights/layer2/weights*
dtype0*
_output_shapes
: 
�
layer2/weights/layer2/weightsHistogramSummary!layer2/weights/layer2/weights/taglayer2/weights/W/read*
T0*
_output_shapes
: 
h
layer2/biases/zerosConst*
valueB*    *
dtype0*
_output_shapes

:
X
layer2/biases/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *���=
k
layer2/biases/addAddlayer2/biases/zeroslayer2/biases/add/y*
T0*
_output_shapes

:
�
layer2/biases/b
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
layer2/biases/b/AssignAssignlayer2/biases/blayer2/biases/add*
use_locking(*
T0*"
_class
loc:@layer2/biases/b*
validate_shape(*
_output_shapes

:
~
layer2/biases/b/readIdentitylayer2/biases/b*
T0*"
_class
loc:@layer2/biases/b*
_output_shapes

:
{
layer2/biases/layer2/biases/tagConst*,
value#B! Blayer2/biases/layer2/biases*
dtype0*
_output_shapes
: 
�
layer2/biases/layer2/biasesHistogramSummarylayer2/biases/layer2/biases/taglayer2/biases/b/read*
T0*
_output_shapes
: 
�
layer2/y/MatMulMatMullayer1/Relulayer2/weights/W/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
l
layer2/y/AddAddlayer2/y/MatMullayer2/biases/b/read*'
_output_shapes
:���������*
T0
_
loss/subSubinputs/y_inputlayer2/y/Add*
T0*'
_output_shapes
:���������
Q
loss/SquareSquareloss/sub*
T0*'
_output_shapes
:���������
d
loss/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
loss/SumSumloss/Squareloss/Sum/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
	loss/MeanMeanloss/Sum
loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
loss/loss/tagsConst*
valueB B	loss/loss*
dtype0*
_output_shapes
: 
V
	loss/lossScalarSummaryloss/loss/tags	loss/Mean*
_output_shapes
: *
T0
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
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
l
$train/gradients/loss/Mean_grad/ShapeShapeloss/Sum*
_output_shapes
:*
T0*
out_type0
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
n
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/Sum*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
p
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:���������
n
#train/gradients/loss/Sum_grad/ShapeShapeloss/Square*
T0*
out_type0*
_output_shapes
:
�
"train/gradients/loss/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B :
�
!train/gradients/loss/Sum_grad/addAddloss/Sum/reduction_indices"train/gradients/loss/Sum_grad/Size*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
!train/gradients/loss/Sum_grad/modFloorMod!train/gradients/loss/Sum_grad/add"train/gradients/loss/Sum_grad/Size*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
%train/gradients/loss/Sum_grad/Shape_1Const*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
valueB:*
dtype0*
_output_shapes
:
�
)train/gradients/loss/Sum_grad/range/startConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
)train/gradients/loss/Sum_grad/range/deltaConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
#train/gradients/loss/Sum_grad/rangeRange)train/gradients/loss/Sum_grad/range/start"train/gradients/loss/Sum_grad/Size)train/gradients/loss/Sum_grad/range/delta*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:*

Tidx0
�
(train/gradients/loss/Sum_grad/Fill/valueConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
"train/gradients/loss/Sum_grad/FillFill%train/gradients/loss/Sum_grad/Shape_1(train/gradients/loss/Sum_grad/Fill/value*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*

index_type0*
_output_shapes
:
�
+train/gradients/loss/Sum_grad/DynamicStitchDynamicStitch#train/gradients/loss/Sum_grad/range!train/gradients/loss/Sum_grad/mod#train/gradients/loss/Sum_grad/Shape"train/gradients/loss/Sum_grad/Fill*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
N*#
_output_shapes
:���������
�
'train/gradients/loss/Sum_grad/Maximum/yConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
%train/gradients/loss/Sum_grad/MaximumMaximum+train/gradients/loss/Sum_grad/DynamicStitch'train/gradients/loss/Sum_grad/Maximum/y*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*#
_output_shapes
:���������
�
&train/gradients/loss/Sum_grad/floordivFloorDiv#train/gradients/loss/Sum_grad/Shape%train/gradients/loss/Sum_grad/Maximum*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
%train/gradients/loss/Sum_grad/ReshapeReshape&train/gradients/loss/Mean_grad/truediv+train/gradients/loss/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
"train/gradients/loss/Sum_grad/TileTile%train/gradients/loss/Sum_grad/Reshape&train/gradients/loss/Sum_grad/floordiv*
T0*'
_output_shapes
:���������*

Tmultiples0
�
&train/gradients/loss/Square_grad/ConstConst#^train/gradients/loss/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
$train/gradients/loss/Square_grad/MulMulloss/sub&train/gradients/loss/Square_grad/Const*
T0*'
_output_shapes
:���������
�
&train/gradients/loss/Square_grad/Mul_1Mul"train/gradients/loss/Sum_grad/Tile$train/gradients/loss/Square_grad/Mul*'
_output_shapes
:���������*
T0
q
#train/gradients/loss/sub_grad/ShapeShapeinputs/y_input*
T0*
out_type0*
_output_shapes
:
q
%train/gradients/loss/sub_grad/Shape_1Shapelayer2/y/Add*
_output_shapes
:*
T0*
out_type0
�
3train/gradients/loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/sub_grad/Shape%train/gradients/loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
!train/gradients/loss/sub_grad/SumSum&train/gradients/loss/Square_grad/Mul_13train/gradients/loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
%train/gradients/loss/sub_grad/ReshapeReshape!train/gradients/loss/sub_grad/Sum#train/gradients/loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
#train/gradients/loss/sub_grad/Sum_1Sum&train/gradients/loss/Square_grad/Mul_15train/gradients/loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
p
!train/gradients/loss/sub_grad/NegNeg#train/gradients/loss/sub_grad/Sum_1*
_output_shapes
:*
T0
�
'train/gradients/loss/sub_grad/Reshape_1Reshape!train/gradients/loss/sub_grad/Neg%train/gradients/loss/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
.train/gradients/loss/sub_grad/tuple/group_depsNoOp&^train/gradients/loss/sub_grad/Reshape(^train/gradients/loss/sub_grad/Reshape_1
�
6train/gradients/loss/sub_grad/tuple/control_dependencyIdentity%train/gradients/loss/sub_grad/Reshape/^train/gradients/loss/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*8
_class.
,*loc:@train/gradients/loss/sub_grad/Reshape
�
8train/gradients/loss/sub_grad/tuple/control_dependency_1Identity'train/gradients/loss/sub_grad/Reshape_1/^train/gradients/loss/sub_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/sub_grad/Reshape_1*'
_output_shapes
:���������
v
'train/gradients/layer2/y/Add_grad/ShapeShapelayer2/y/MatMul*
_output_shapes
:*
T0*
out_type0
z
)train/gradients/layer2/y/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
�
7train/gradients/layer2/y/Add_grad/BroadcastGradientArgsBroadcastGradientArgs'train/gradients/layer2/y/Add_grad/Shape)train/gradients/layer2/y/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
%train/gradients/layer2/y/Add_grad/SumSum8train/gradients/loss/sub_grad/tuple/control_dependency_17train/gradients/layer2/y/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/layer2/y/Add_grad/ReshapeReshape%train/gradients/layer2/y/Add_grad/Sum'train/gradients/layer2/y/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
'train/gradients/layer2/y/Add_grad/Sum_1Sum8train/gradients/loss/sub_grad/tuple/control_dependency_19train/gradients/layer2/y/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
+train/gradients/layer2/y/Add_grad/Reshape_1Reshape'train/gradients/layer2/y/Add_grad/Sum_1)train/gradients/layer2/y/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
2train/gradients/layer2/y/Add_grad/tuple/group_depsNoOp*^train/gradients/layer2/y/Add_grad/Reshape,^train/gradients/layer2/y/Add_grad/Reshape_1
�
:train/gradients/layer2/y/Add_grad/tuple/control_dependencyIdentity)train/gradients/layer2/y/Add_grad/Reshape3^train/gradients/layer2/y/Add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/layer2/y/Add_grad/Reshape*'
_output_shapes
:���������
�
<train/gradients/layer2/y/Add_grad/tuple/control_dependency_1Identity+train/gradients/layer2/y/Add_grad/Reshape_13^train/gradients/layer2/y/Add_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/layer2/y/Add_grad/Reshape_1*
_output_shapes

:
�
+train/gradients/layer2/y/MatMul_grad/MatMulMatMul:train/gradients/layer2/y/Add_grad/tuple/control_dependencylayer2/weights/W/read*
transpose_b(*
T0*'
_output_shapes
:���������
*
transpose_a( 
�
-train/gradients/layer2/y/MatMul_grad/MatMul_1MatMullayer1/Relu:train/gradients/layer2/y/Add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
5train/gradients/layer2/y/MatMul_grad/tuple/group_depsNoOp,^train/gradients/layer2/y/MatMul_grad/MatMul.^train/gradients/layer2/y/MatMul_grad/MatMul_1
�
=train/gradients/layer2/y/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/layer2/y/MatMul_grad/MatMul6^train/gradients/layer2/y/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/layer2/y/MatMul_grad/MatMul*'
_output_shapes
:���������

�
?train/gradients/layer2/y/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/layer2/y/MatMul_grad/MatMul_16^train/gradients/layer2/y/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/layer2/y/MatMul_grad/MatMul_1*
_output_shapes

:

�
)train/gradients/layer1/Relu_grad/ReluGradReluGrad=train/gradients/layer2/y/MatMul_grad/tuple/control_dependencylayer1/Relu*'
_output_shapes
:���������
*
T0
v
'train/gradients/layer1/y/Add_grad/ShapeShapelayer1/y/MatMul*
T0*
out_type0*
_output_shapes
:
z
)train/gradients/layer1/y/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
�
7train/gradients/layer1/y/Add_grad/BroadcastGradientArgsBroadcastGradientArgs'train/gradients/layer1/y/Add_grad/Shape)train/gradients/layer1/y/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%train/gradients/layer1/y/Add_grad/SumSum)train/gradients/layer1/Relu_grad/ReluGrad7train/gradients/layer1/y/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
)train/gradients/layer1/y/Add_grad/ReshapeReshape%train/gradients/layer1/y/Add_grad/Sum'train/gradients/layer1/y/Add_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
'train/gradients/layer1/y/Add_grad/Sum_1Sum)train/gradients/layer1/Relu_grad/ReluGrad9train/gradients/layer1/y/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
+train/gradients/layer1/y/Add_grad/Reshape_1Reshape'train/gradients/layer1/y/Add_grad/Sum_1)train/gradients/layer1/y/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
2train/gradients/layer1/y/Add_grad/tuple/group_depsNoOp*^train/gradients/layer1/y/Add_grad/Reshape,^train/gradients/layer1/y/Add_grad/Reshape_1
�
:train/gradients/layer1/y/Add_grad/tuple/control_dependencyIdentity)train/gradients/layer1/y/Add_grad/Reshape3^train/gradients/layer1/y/Add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/layer1/y/Add_grad/Reshape*'
_output_shapes
:���������

�
<train/gradients/layer1/y/Add_grad/tuple/control_dependency_1Identity+train/gradients/layer1/y/Add_grad/Reshape_13^train/gradients/layer1/y/Add_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/layer1/y/Add_grad/Reshape_1*
_output_shapes

:

�
+train/gradients/layer1/y/MatMul_grad/MatMulMatMul:train/gradients/layer1/y/Add_grad/tuple/control_dependencylayer1/weights/W/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
-train/gradients/layer1/y/MatMul_grad/MatMul_1MatMulinputs/x_input:train/gradients/layer1/y/Add_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
�
5train/gradients/layer1/y/MatMul_grad/tuple/group_depsNoOp,^train/gradients/layer1/y/MatMul_grad/MatMul.^train/gradients/layer1/y/MatMul_grad/MatMul_1
�
=train/gradients/layer1/y/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/layer1/y/MatMul_grad/MatMul6^train/gradients/layer1/y/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/layer1/y/MatMul_grad/MatMul*'
_output_shapes
:���������
�
?train/gradients/layer1/y/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/layer1/y/MatMul_grad/MatMul_16^train/gradients/layer1/y/MatMul_grad/tuple/group_deps*
_output_shapes

:
*
T0*@
_class6
42loc:@train/gradients/layer1/y/MatMul_grad/MatMul_1
h
#train/GradientDescent/learning_rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
Btrain/GradientDescent/update_layer1/weights/W/ApplyGradientDescentApplyGradientDescentlayer1/weights/W#train/GradientDescent/learning_rate?train/gradients/layer1/y/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer1/weights/W*
_output_shapes

:

�
Atrain/GradientDescent/update_layer1/biases/b/ApplyGradientDescentApplyGradientDescentlayer1/biases/b#train/GradientDescent/learning_rate<train/gradients/layer1/y/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer1/biases/b*
_output_shapes

:

�
Btrain/GradientDescent/update_layer2/weights/W/ApplyGradientDescentApplyGradientDescentlayer2/weights/W#train/GradientDescent/learning_rate?train/gradients/layer2/y/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer2/weights/W*
_output_shapes

:

�
Atrain/GradientDescent/update_layer2/biases/b/ApplyGradientDescentApplyGradientDescentlayer2/biases/b#train/GradientDescent/learning_rate<train/gradients/layer2/y/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer2/biases/b*
_output_shapes

:
�
train/GradientDescentNoOpB^train/GradientDescent/update_layer1/biases/b/ApplyGradientDescentC^train/GradientDescent/update_layer1/weights/W/ApplyGradientDescentB^train/GradientDescent/update_layer2/biases/b/ApplyGradientDescentC^train/GradientDescent/update_layer2/weights/W/ApplyGradientDescent
�
Merge/MergeSummaryMergeSummarylayer1/weights/layer1/weightslayer1/biases/layer1/biaseslayer1/layer1/outputslayer2/weights/layer2/weightslayer2/biases/layer2/biases	loss/loss*
N*
_output_shapes
: "����k      �?�	���I�'�AJ��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyGradientDescent
var"T�

alpha"T

delta"T
out"T�" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
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
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
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

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
1
Square
x"T
y"T"
Ttype:

2	
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
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.9.02v1.9.0-0-g25c197e023��
q
inputs/x_inputPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
q
inputs/y_inputPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
s
"layer1/weights/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
f
!layer1/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
#layer1/weights/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
1layer1/weights/random_normal/RandomStandardNormalRandomStandardNormal"layer1/weights/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
 layer1/weights/random_normal/mulMul1layer1/weights/random_normal/RandomStandardNormal#layer1/weights/random_normal/stddev*
T0*
_output_shapes

:

�
layer1/weights/random_normalAdd layer1/weights/random_normal/mul!layer1/weights/random_normal/mean*
T0*
_output_shapes

:

�
layer1/weights/W
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
layer1/weights/W/AssignAssignlayer1/weights/Wlayer1/weights/random_normal*
use_locking(*
T0*#
_class
loc:@layer1/weights/W*
validate_shape(*
_output_shapes

:

�
layer1/weights/W/readIdentitylayer1/weights/W*
T0*#
_class
loc:@layer1/weights/W*
_output_shapes

:


!layer1/weights/layer1/weights/tagConst*.
value%B# Blayer1/weights/layer1/weights*
dtype0*
_output_shapes
: 
�
layer1/weights/layer1/weightsHistogramSummary!layer1/weights/layer1/weights/taglayer1/weights/W/read*
T0*
_output_shapes
: 
h
layer1/biases/zerosConst*
valueB
*    *
dtype0*
_output_shapes

:

X
layer1/biases/add/yConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
k
layer1/biases/addAddlayer1/biases/zeroslayer1/biases/add/y*
_output_shapes

:
*
T0
�
layer1/biases/b
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
�
layer1/biases/b/AssignAssignlayer1/biases/blayer1/biases/add*
use_locking(*
T0*"
_class
loc:@layer1/biases/b*
validate_shape(*
_output_shapes

:

~
layer1/biases/b/readIdentitylayer1/biases/b*
T0*"
_class
loc:@layer1/biases/b*
_output_shapes

:

{
layer1/biases/layer1/biases/tagConst*,
value#B! Blayer1/biases/layer1/biases*
dtype0*
_output_shapes
: 
�
layer1/biases/layer1/biasesHistogramSummarylayer1/biases/layer1/biases/taglayer1/biases/b/read*
T0*
_output_shapes
: 
�
layer1/y/MatMulMatMulinputs/x_inputlayer1/weights/W/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
l
layer1/y/AddAddlayer1/y/MatMullayer1/biases/b/read*'
_output_shapes
:���������
*
T0
S
layer1/ReluRelulayer1/y/Add*
T0*'
_output_shapes
:���������

o
layer1/layer1/outputs/tagConst*&
valueB Blayer1/layer1/outputs*
dtype0*
_output_shapes
: 
r
layer1/layer1/outputsHistogramSummarylayer1/layer1/outputs/taglayer1/Relu*
T0*
_output_shapes
: 
s
"layer2/weights/random_normal/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
f
!layer2/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
#layer2/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
1layer2/weights/random_normal/RandomStandardNormalRandomStandardNormal"layer2/weights/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
 layer2/weights/random_normal/mulMul1layer2/weights/random_normal/RandomStandardNormal#layer2/weights/random_normal/stddev*
T0*
_output_shapes

:

�
layer2/weights/random_normalAdd layer2/weights/random_normal/mul!layer2/weights/random_normal/mean*
T0*
_output_shapes

:

�
layer2/weights/W
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
layer2/weights/W/AssignAssignlayer2/weights/Wlayer2/weights/random_normal*
T0*#
_class
loc:@layer2/weights/W*
validate_shape(*
_output_shapes

:
*
use_locking(
�
layer2/weights/W/readIdentitylayer2/weights/W*
T0*#
_class
loc:@layer2/weights/W*
_output_shapes

:


!layer2/weights/layer2/weights/tagConst*.
value%B# Blayer2/weights/layer2/weights*
dtype0*
_output_shapes
: 
�
layer2/weights/layer2/weightsHistogramSummary!layer2/weights/layer2/weights/taglayer2/weights/W/read*
T0*
_output_shapes
: 
h
layer2/biases/zerosConst*
valueB*    *
dtype0*
_output_shapes

:
X
layer2/biases/add/yConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
k
layer2/biases/addAddlayer2/biases/zeroslayer2/biases/add/y*
T0*
_output_shapes

:
�
layer2/biases/b
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
layer2/biases/b/AssignAssignlayer2/biases/blayer2/biases/add*
T0*"
_class
loc:@layer2/biases/b*
validate_shape(*
_output_shapes

:*
use_locking(
~
layer2/biases/b/readIdentitylayer2/biases/b*
_output_shapes

:*
T0*"
_class
loc:@layer2/biases/b
{
layer2/biases/layer2/biases/tagConst*,
value#B! Blayer2/biases/layer2/biases*
dtype0*
_output_shapes
: 
�
layer2/biases/layer2/biasesHistogramSummarylayer2/biases/layer2/biases/taglayer2/biases/b/read*
T0*
_output_shapes
: 
�
layer2/y/MatMulMatMullayer1/Relulayer2/weights/W/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
l
layer2/y/AddAddlayer2/y/MatMullayer2/biases/b/read*'
_output_shapes
:���������*
T0
_
loss/subSubinputs/y_inputlayer2/y/Add*'
_output_shapes
:���������*
T0
Q
loss/SquareSquareloss/sub*
T0*'
_output_shapes
:���������
d
loss/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
loss/SumSumloss/Squareloss/Sum/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
	loss/MeanMeanloss/Sum
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
loss/loss/tagsConst*
valueB B	loss/loss*
dtype0*
_output_shapes
: 
V
	loss/lossScalarSummaryloss/loss/tags	loss/Mean*
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
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
l
$train/gradients/loss/Mean_grad/ShapeShapeloss/Sum*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
n
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/Sum*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:���������
n
#train/gradients/loss/Sum_grad/ShapeShapeloss/Square*
_output_shapes
:*
T0*
out_type0
�
"train/gradients/loss/Sum_grad/SizeConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
!train/gradients/loss/Sum_grad/addAddloss/Sum/reduction_indices"train/gradients/loss/Sum_grad/Size*
_output_shapes
:*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
!train/gradients/loss/Sum_grad/modFloorMod!train/gradients/loss/Sum_grad/add"train/gradients/loss/Sum_grad/Size*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
%train/gradients/loss/Sum_grad/Shape_1Const*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
valueB:*
dtype0*
_output_shapes
:
�
)train/gradients/loss/Sum_grad/range/startConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
)train/gradients/loss/Sum_grad/range/deltaConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
#train/gradients/loss/Sum_grad/rangeRange)train/gradients/loss/Sum_grad/range/start"train/gradients/loss/Sum_grad/Size)train/gradients/loss/Sum_grad/range/delta*

Tidx0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
(train/gradients/loss/Sum_grad/Fill/valueConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
"train/gradients/loss/Sum_grad/FillFill%train/gradients/loss/Sum_grad/Shape_1(train/gradients/loss/Sum_grad/Fill/value*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*

index_type0*
_output_shapes
:
�
+train/gradients/loss/Sum_grad/DynamicStitchDynamicStitch#train/gradients/loss/Sum_grad/range!train/gradients/loss/Sum_grad/mod#train/gradients/loss/Sum_grad/Shape"train/gradients/loss/Sum_grad/Fill*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
N*#
_output_shapes
:���������
�
'train/gradients/loss/Sum_grad/Maximum/yConst*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
%train/gradients/loss/Sum_grad/MaximumMaximum+train/gradients/loss/Sum_grad/DynamicStitch'train/gradients/loss/Sum_grad/Maximum/y*#
_output_shapes
:���������*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape
�
&train/gradients/loss/Sum_grad/floordivFloorDiv#train/gradients/loss/Sum_grad/Shape%train/gradients/loss/Sum_grad/Maximum*
T0*6
_class,
*(loc:@train/gradients/loss/Sum_grad/Shape*
_output_shapes
:
�
%train/gradients/loss/Sum_grad/ReshapeReshape&train/gradients/loss/Mean_grad/truediv+train/gradients/loss/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
"train/gradients/loss/Sum_grad/TileTile%train/gradients/loss/Sum_grad/Reshape&train/gradients/loss/Sum_grad/floordiv*'
_output_shapes
:���������*

Tmultiples0*
T0
�
&train/gradients/loss/Square_grad/ConstConst#^train/gradients/loss/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
$train/gradients/loss/Square_grad/MulMulloss/sub&train/gradients/loss/Square_grad/Const*
T0*'
_output_shapes
:���������
�
&train/gradients/loss/Square_grad/Mul_1Mul"train/gradients/loss/Sum_grad/Tile$train/gradients/loss/Square_grad/Mul*
T0*'
_output_shapes
:���������
q
#train/gradients/loss/sub_grad/ShapeShapeinputs/y_input*
T0*
out_type0*
_output_shapes
:
q
%train/gradients/loss/sub_grad/Shape_1Shapelayer2/y/Add*
_output_shapes
:*
T0*
out_type0
�
3train/gradients/loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/sub_grad/Shape%train/gradients/loss/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
!train/gradients/loss/sub_grad/SumSum&train/gradients/loss/Square_grad/Mul_13train/gradients/loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
%train/gradients/loss/sub_grad/ReshapeReshape!train/gradients/loss/sub_grad/Sum#train/gradients/loss/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
#train/gradients/loss/sub_grad/Sum_1Sum&train/gradients/loss/Square_grad/Mul_15train/gradients/loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
p
!train/gradients/loss/sub_grad/NegNeg#train/gradients/loss/sub_grad/Sum_1*
T0*
_output_shapes
:
�
'train/gradients/loss/sub_grad/Reshape_1Reshape!train/gradients/loss/sub_grad/Neg%train/gradients/loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
.train/gradients/loss/sub_grad/tuple/group_depsNoOp&^train/gradients/loss/sub_grad/Reshape(^train/gradients/loss/sub_grad/Reshape_1
�
6train/gradients/loss/sub_grad/tuple/control_dependencyIdentity%train/gradients/loss/sub_grad/Reshape/^train/gradients/loss/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*8
_class.
,*loc:@train/gradients/loss/sub_grad/Reshape
�
8train/gradients/loss/sub_grad/tuple/control_dependency_1Identity'train/gradients/loss/sub_grad/Reshape_1/^train/gradients/loss/sub_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/loss/sub_grad/Reshape_1*'
_output_shapes
:���������
v
'train/gradients/layer2/y/Add_grad/ShapeShapelayer2/y/MatMul*
T0*
out_type0*
_output_shapes
:
z
)train/gradients/layer2/y/Add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
7train/gradients/layer2/y/Add_grad/BroadcastGradientArgsBroadcastGradientArgs'train/gradients/layer2/y/Add_grad/Shape)train/gradients/layer2/y/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%train/gradients/layer2/y/Add_grad/SumSum8train/gradients/loss/sub_grad/tuple/control_dependency_17train/gradients/layer2/y/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
)train/gradients/layer2/y/Add_grad/ReshapeReshape%train/gradients/layer2/y/Add_grad/Sum'train/gradients/layer2/y/Add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
'train/gradients/layer2/y/Add_grad/Sum_1Sum8train/gradients/loss/sub_grad/tuple/control_dependency_19train/gradients/layer2/y/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
+train/gradients/layer2/y/Add_grad/Reshape_1Reshape'train/gradients/layer2/y/Add_grad/Sum_1)train/gradients/layer2/y/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
2train/gradients/layer2/y/Add_grad/tuple/group_depsNoOp*^train/gradients/layer2/y/Add_grad/Reshape,^train/gradients/layer2/y/Add_grad/Reshape_1
�
:train/gradients/layer2/y/Add_grad/tuple/control_dependencyIdentity)train/gradients/layer2/y/Add_grad/Reshape3^train/gradients/layer2/y/Add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/layer2/y/Add_grad/Reshape*'
_output_shapes
:���������
�
<train/gradients/layer2/y/Add_grad/tuple/control_dependency_1Identity+train/gradients/layer2/y/Add_grad/Reshape_13^train/gradients/layer2/y/Add_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/layer2/y/Add_grad/Reshape_1*
_output_shapes

:
�
+train/gradients/layer2/y/MatMul_grad/MatMulMatMul:train/gradients/layer2/y/Add_grad/tuple/control_dependencylayer2/weights/W/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(
�
-train/gradients/layer2/y/MatMul_grad/MatMul_1MatMullayer1/Relu:train/gradients/layer2/y/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
�
5train/gradients/layer2/y/MatMul_grad/tuple/group_depsNoOp,^train/gradients/layer2/y/MatMul_grad/MatMul.^train/gradients/layer2/y/MatMul_grad/MatMul_1
�
=train/gradients/layer2/y/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/layer2/y/MatMul_grad/MatMul6^train/gradients/layer2/y/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*>
_class4
20loc:@train/gradients/layer2/y/MatMul_grad/MatMul
�
?train/gradients/layer2/y/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/layer2/y/MatMul_grad/MatMul_16^train/gradients/layer2/y/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/layer2/y/MatMul_grad/MatMul_1*
_output_shapes

:

�
)train/gradients/layer1/Relu_grad/ReluGradReluGrad=train/gradients/layer2/y/MatMul_grad/tuple/control_dependencylayer1/Relu*
T0*'
_output_shapes
:���������

v
'train/gradients/layer1/y/Add_grad/ShapeShapelayer1/y/MatMul*
T0*
out_type0*
_output_shapes
:
z
)train/gradients/layer1/y/Add_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
�
7train/gradients/layer1/y/Add_grad/BroadcastGradientArgsBroadcastGradientArgs'train/gradients/layer1/y/Add_grad/Shape)train/gradients/layer1/y/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%train/gradients/layer1/y/Add_grad/SumSum)train/gradients/layer1/Relu_grad/ReluGrad7train/gradients/layer1/y/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
)train/gradients/layer1/y/Add_grad/ReshapeReshape%train/gradients/layer1/y/Add_grad/Sum'train/gradients/layer1/y/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
'train/gradients/layer1/y/Add_grad/Sum_1Sum)train/gradients/layer1/Relu_grad/ReluGrad9train/gradients/layer1/y/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
+train/gradients/layer1/y/Add_grad/Reshape_1Reshape'train/gradients/layer1/y/Add_grad/Sum_1)train/gradients/layer1/y/Add_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
�
2train/gradients/layer1/y/Add_grad/tuple/group_depsNoOp*^train/gradients/layer1/y/Add_grad/Reshape,^train/gradients/layer1/y/Add_grad/Reshape_1
�
:train/gradients/layer1/y/Add_grad/tuple/control_dependencyIdentity)train/gradients/layer1/y/Add_grad/Reshape3^train/gradients/layer1/y/Add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/layer1/y/Add_grad/Reshape*'
_output_shapes
:���������

�
<train/gradients/layer1/y/Add_grad/tuple/control_dependency_1Identity+train/gradients/layer1/y/Add_grad/Reshape_13^train/gradients/layer1/y/Add_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/layer1/y/Add_grad/Reshape_1*
_output_shapes

:

�
+train/gradients/layer1/y/MatMul_grad/MatMulMatMul:train/gradients/layer1/y/Add_grad/tuple/control_dependencylayer1/weights/W/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
-train/gradients/layer1/y/MatMul_grad/MatMul_1MatMulinputs/x_input:train/gradients/layer1/y/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
�
5train/gradients/layer1/y/MatMul_grad/tuple/group_depsNoOp,^train/gradients/layer1/y/MatMul_grad/MatMul.^train/gradients/layer1/y/MatMul_grad/MatMul_1
�
=train/gradients/layer1/y/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/layer1/y/MatMul_grad/MatMul6^train/gradients/layer1/y/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/layer1/y/MatMul_grad/MatMul*'
_output_shapes
:���������
�
?train/gradients/layer1/y/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/layer1/y/MatMul_grad/MatMul_16^train/gradients/layer1/y/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/layer1/y/MatMul_grad/MatMul_1*
_output_shapes

:

h
#train/GradientDescent/learning_rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
Btrain/GradientDescent/update_layer1/weights/W/ApplyGradientDescentApplyGradientDescentlayer1/weights/W#train/GradientDescent/learning_rate?train/gradients/layer1/y/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*#
_class
loc:@layer1/weights/W
�
Atrain/GradientDescent/update_layer1/biases/b/ApplyGradientDescentApplyGradientDescentlayer1/biases/b#train/GradientDescent/learning_rate<train/gradients/layer1/y/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer1/biases/b*
_output_shapes

:

�
Btrain/GradientDescent/update_layer2/weights/W/ApplyGradientDescentApplyGradientDescentlayer2/weights/W#train/GradientDescent/learning_rate?train/gradients/layer2/y/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer2/weights/W*
_output_shapes

:

�
Atrain/GradientDescent/update_layer2/biases/b/ApplyGradientDescentApplyGradientDescentlayer2/biases/b#train/GradientDescent/learning_rate<train/gradients/layer2/y/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer2/biases/b*
_output_shapes

:
�
train/GradientDescentNoOpB^train/GradientDescent/update_layer1/biases/b/ApplyGradientDescentC^train/GradientDescent/update_layer1/weights/W/ApplyGradientDescentB^train/GradientDescent/update_layer2/biases/b/ApplyGradientDescentC^train/GradientDescent/update_layer2/weights/W/ApplyGradientDescent
�
Merge/MergeSummaryMergeSummarylayer1/weights/layer1/weightslayer1/biases/layer1/biaseslayer1/layer1/outputslayer2/weights/layer2/weightslayer2/biases/layer2/biases	loss/loss*
N*
_output_shapes
: ""�
	summaries�
�
layer1/weights/layer1/weights:0
layer1/biases/layer1/biases:0
layer1/layer1/outputs:0
layer2/weights/layer2/weights:0
layer2/biases/layer2/biases:0
loss/loss:0"�
trainable_variables��
h
layer1/weights/W:0layer1/weights/W/Assignlayer1/weights/W/read:02layer1/weights/random_normal:08
Z
layer1/biases/b:0layer1/biases/b/Assignlayer1/biases/b/read:02layer1/biases/add:08
h
layer2/weights/W:0layer2/weights/W/Assignlayer2/weights/W/read:02layer2/weights/random_normal:08
Z
layer2/biases/b:0layer2/biases/b/Assignlayer2/biases/b/read:02layer2/biases/add:08"%
train_op

train/GradientDescent"�
	variables��
h
layer1/weights/W:0layer1/weights/W/Assignlayer1/weights/W/read:02layer1/weights/random_normal:08
Z
layer1/biases/b:0layer1/biases/b/Assignlayer1/biases/b/read:02layer1/biases/add:08
h
layer2/weights/W:0layer2/weights/W/Assignlayer2/weights/W/read:02layer2/weights/random_normal:08
Z
layer2/biases/b:0layer2/biases/b/Assignlayer2/biases/b/read:02layer2/biases/add:08�+m|	      ���'	�
�I�'�A*�
�
layer1/weights/layer1/weights*�	   `Gt�   ����?      $@!   *��@)xrͱ�"@2��P�1���cI���+�;$��iZ��2g�G�A�uo�p�W�i�bۿ�^��h�ؿ�QK|:�?�@�"��?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?               @              �?              �?              �?        
�
layer1/biases/layer1/biases*�	   �G��?    Ԭ�?      $@!   �$��?)�}�8S�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      @      @        
�

layer1/layer1/outputs*�
   `<~�?     p�@! ���É@)r�Z��1�@2�        �-���q=a�$��{E?
����G?�lDZrS?<DKc��T?ܗ�SsW?E��{��^?�l�P�`?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             ��@              �?              �?      �?              �?              �?      �?               @      �?      �?      �?       @      �?       @       @      �?      �?      @      @      @      �?       @      @      @      @      @      @      @      @       @      @      @      *@       @      *@      *@      ,@      .@      0@      2@      3@      9@      7@      ;@      ?@      A@     �A@      D@     �E@     �G@      K@     �L@     �M@     �G@      J@      L@     �O@     @Q@     �R@     �P@     �Q@     @S@     �U@     @V@     �S@      N@     �G@     �C@      ?@      (@      ,@      "@        
�
layer2/weights/layer2/weights*�	   `�
 �   @�H�?      $@!  �X�)N8�Ff-@2���tM�ܔ�.�u��S�Fi���+�;$��iZ���K?̿�@�"�ɿ�/�*>��`��a�8��*QH�x�&b՞
�u�������?�iZ�?cI���?�P�1���?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   @�9�?   @�9�?      �?!   @�9�?) �]0t�?2!�����?Ӗ8��s�?�������:              �?        

	loss/lossD>�.E�P
      �"L	�"�I�'�A2*�
�
layer1/weights/layer1/weights*�	    С�   ���?      $@!   ��~@)ZD�9_"@2��P�1���cI���+�;$��iZ��+Se*8�\l�9⿰1%��Z�_���?����?uo�p�?2g�G�A�?������?�iZ�?�P�1���?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?      �?              �?              �?      �?      �?              �?              �?        
�
layer1/biases/layer1/biases*�	   `?l��    u�?      $@!   LІ�?)tʅ����?2���(!�ؼ�%g�cE9���/�*>��`��a�8���/��?�uS��a�?`��a�8�?�/�*>�?���g��?I���?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�Z�_���?����?�������:�              �?              �?              �?              �?              �?               @      �?      �?              �?        
�

layer1/layer1/outputs*�
   `�d�?     p�@!  `8Y�@)jr?w{�@2�        �-���q=��%�V6?uܬ�@8?
����G?�qU���I?ܗ�SsW?��bB�SY?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             �@              �?               @              �?              �?              �?              �?      �?              �?      @      �?      �?      �?       @              @      �?       @      �?       @      @       @      @      @      @      @      �?      "@      @      @      @       @      "@      "@      &@      &@      (@      1@      ,@      2@      1@      5@      6@      6@      <@      >@      A@      A@      D@      E@     �H@     �J@     �L@     @P@     �Q@     �R@     �U@     �V@      X@     �S@     @S@     �R@     @R@     �P@     �D@      G@      C@      C@      >@      *@      ,@      @        
�
layer2/weights/layer2/weights*�	    ����   @4��?      $@!   a�C�)vq]�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A��uS��a���/�����(!�ؼ?!�����?yD$��?�QK|:�?�iZ�?+�;$�?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   `��?   `��?      �?!   `��?)@4�9'q?2����iH�?��]$A�?�������:              �?        

	loss/lossw1<?
�� 
      �(#�	g�I�'�Ad*�
�
layer1/weights/layer1/weights*�	   `���   @y��?      $@!   8�@)>ʋ�g�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8�\l�9⿰1%�_&A�o��?�Ca�G��?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?        
�
layer1/biases/layer1/biases*�	   @3�Ŀ   �A��?      $@!  ���t�?)��6g�?2�yD$�ſ�?>8s2ÿ��(!�ؼ�%g�cE9��IcD���L?k�1^�sO?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?_&A�o��?�Ca�G��?�������:�              �?              �?              �?              �?      �?              �?      �?      �?      �?              �?        
�

layer1/layer1/outputs*�	   ��w�?     p�@! p����@)��4���@2�        �-���q=8K�ߝ�>�h���`�>IcD���L?k�1^�sO?�m9�H�[?E��{��^?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             ��@              �?              �?              �?              �?       @      �?       @               @              �?       @      @      �?       @              @      @      �?      @      @      @      @      @      @      @      @      @      @       @      "@      &@      "@      *@      &@      1@      1@      0@      6@      4@      7@      9@      >@      @@      A@     �B@     �E@     �F@     �J@     �K@      N@     �Q@      R@      T@     �V@     @Y@     �W@      U@     �P@     �Q@     �E@     �D@     �F@     �C@     �B@      =@      4@      ,@      @        
�
layer2/weights/layer2/weights*�	    *���   @j��?      $@!  `v%L�)� <*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迄m9�H�[���bB�SY�Ӗ8��s�?�?>8s2�?�@�"��?�K?�?�iZ�?+�;$�?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   ����?   ����?      �?!   ����?) �j��{`?2�/�*>�?�g���w�?�������:              �?        

	loss/loss�3�;�jtFA
      Wm;�	L�I�'�A�*�
�
layer1/weights/layer1/weights*�	    m��   @X��?      $@!   �p�@)�X��Γ"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?uo�p�?2g�G�A�?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?               @              �?              �?              �?        
�
layer1/biases/layer1/biases*�	   �Tɿ   `.��?      $@!  @���?)����af�?2��@�"�ɿ�QK|:ǿ�?>8s2ÿӖ8��s���N�W�m�ߤ�(g%k��uS��a�?`��a�8�?���g��?I���?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?_&A�o��?�Ca�G��?�������:�              �?              �?              �?              �?              �?               @               @              �?        
�

layer1/layer1/outputs*�
   `"�?     p�@! �pZ͇@)��!y+��@2�        �-���q=���#@?�!�A?IcD���L?k�1^�sO?nK���LQ?�lDZrS?�m9�H�[?E��{��^?�l�P�`?Tw��Nof?P}���h?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              �@              �?              �?              �?              �?      �?              �?               @      �?      �?      �?       @               @      �?      @       @      �?      @       @      @      @      @      @      @      @      @      @      @      @      @      "@      "@      "@      *@      (@      (@      0@      0@      1@      4@      4@      8@      9@      <@     �@@     �A@      B@     �D@     �F@     �I@      L@     �M@     @Q@      R@      T@     �V@     �W@     @W@     �V@      Q@      P@      C@     �C@     �G@      C@     �B@      >@      7@      ,@      @        
�
layer2/weights/layer2/weights*�	    x���   �m��?      $@!  ��@��)&H��Z[*@2�ܔ�.�u��S�Fi���yL������������2g�G�A�eiS�m�?#�+(�ŉ?Ӗ8��s�?�?>8s2�?�K?�?�Z�_���?�iZ�?+�;$�?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   ���?   ���?      �?!   ���?)@�b�O[?2`��a�8�?�/�*>�?�������:              �?        

	loss/loss��;V�xq
      k{f;	4�I�'�A�*�
�
layer1/weights/layer1/weights*�	    ���   �8��?      $@!   tq�@)ƞ����"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?        
�
layer1/biases/layer1/biases*�	    �̿   � ��?      $@!  ����?)���E��?2��Z�_��ο�K?̿�QK|:ǿyD$�ſ�!�A?�T���C?�/��?�uS��a�?���g��?I���?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?_&A�o��?�Ca�G��?�������:�              �?              �?              �?              �?              �?               @              �?      �?              �?        
�

layer1/layer1/outputs*�
   ����?     p�@! ���n��@)�߸a}�@2�        �-���q=;�"�q�>['�?��>��82?�u�w74?�qU���I?IcD���L?�lDZrS?<DKc��T?ܗ�SsW?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             �@              �?              �?               @              �?      �?              �?      �?      �?               @              @              �?       @      �?       @      @      �?      @      @       @      �?      @      @      @      @       @      @       @      @      $@       @       @      &@      &@      *@      (@      .@      1@      1@      3@      4@      8@      :@      =@      >@     �A@     �B@      E@     �F@     �I@      K@      N@     �P@     @R@      T@      W@     @U@     �W@     @W@     �P@      N@      C@     �C@      G@      C@      C@      >@      9@      ,@      @        
�
layer2/weights/layer2/weights*�	   �����   ����?      $@!   ���)�u�Eo*@2�ܔ�.�u��S�Fi���yL������������2g�G�A�^�S���?�"�uԖ?�?>8s2�?yD$��?�Z�_���?����?�iZ�?+�;$�?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	    �Z�?    �Z�?      �?!    �Z�?)@Db-��\?2`��a�8�?�/�*>�?�������:              �?        

	loss/loss�ދ;js��
      U���	c7�I�'�A�*�
�
layer1/weights/layer1/weights*�	   ����   ����?      $@!   ���@)���A��"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?        
�
layer1/biases/layer1/biases*�	   @��ο   `���?      $@!   E�?)T�r����?2��Z�_��ο�K?̿�@�"�ɿuWy��r?hyO�s?�/��?�uS��a�?���g��?I���?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?!�����?Ӗ8��s�?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?              �?              �?      �?      �?              �?              �?        
�

layer1/layer1/outputs*�
    ��?     p�@! ��<I��@)�˳��@2�        �-���q=�uE����>�f����>a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             4�@              �?              �?              �?              �?              �?               @              �?      �?               @              �?      �?      �?       @      �?      @      �?      �?       @      @      @      @              @      @      @       @      @      @      @      @       @      @       @      ,@      "@      &@      .@      ,@      0@      1@      4@      3@      8@      8@      >@      ?@      A@     �B@      E@     �G@      I@      K@     �N@      Q@      R@     @T@     @U@     @U@      X@     �W@     �P@     �K@     �B@     �C@     �G@      C@     �B@      >@      ;@      ,@      @        
�
layer2/weights/layer2/weights*�	   `����   ����?      $@!   yV�)��Rj�}*@2�ܔ�.�u��S�Fi���yL������������2g�G�A过v��ab�?�/��?�?>8s2�?yD$��?�Z�_���?����?�iZ�?+�;$�?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   �{��?   �{��?      �?!   �{��?) ���8`?2�/�*>�?�g���w�?�������:              �?        

	loss/loss��;$62�
      Uɗ�	.>�I�'�A�*�
�
layer1/weights/layer1/weights*�	   @���   ���?      $@!   ��@)���(<�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	    �8п   ��?      $@!   �ap�?)�76��?2����ѿ�Z�_��ο�K?̿>	� �?����=��?�uS��a�?`��a�8�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?!�����?Ӗ8��s�?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?        
�

layer1/layer1/outputs*�
   �>}�?     p�@! @Z8��@)��� �@2�        �-���q=
�/eq
�>;�"�q�>��%>��:?d�\D�X=?���#@?�qU���I?IcD���L?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             @�@              �?              �?      �?              �?               @              �?              �?              �?      �?              @      �?       @      �?       @       @      �?      �?      @      @      �?      @      @      @      @      @      @      @       @      @      @      "@      @      (@      $@      *@      ,@      ,@      .@      2@      2@      7@      6@      ;@      <@      ?@     �@@     �C@      D@     �G@      I@     �K@      O@     @P@     �R@      T@      T@     @V@     @W@     �W@     @Q@      J@      C@      D@     �F@     �C@     �B@      =@      <@      .@      @        
�
layer2/weights/layer2/weights*�	   @6���   ����?      $@!   T���)4sh�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A��uS��a�?`��a�8�?yD$��?�QK|:�?�Z�_���?����?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	    ��?    ��?      �?!    ��?) 8�mb?2�/�*>�?�g���w�?�������:              �?        

	loss/loss��p;�K��1
      j�
�	�L�I�'�A�*�
�
layer1/weights/layer1/weights*�	   @���    ���?      $@!   �ϲ@)�,�g��"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   ��п   �,'�?      $@!   ��7�?)�4鷝��?2����ѿ�Z�_��οeiS�m�?#�+(�ŉ?`��a�8�?�/�*>�?���g��?I���?��]$A�?�{ �ǳ�?8/�C�ַ?%g�cE9�?Ӗ8��s�?�?>8s2�?_&A�o��?�Ca�G��?�������:�               @              �?              �?              �?              �?               @              �?              �?        
�

layer1/layer1/outputs*�
   �]z�?     p�@!  |�쯇@)��BZ�+�@2�        �-���q=��d�r?�5�i}1?U�4@@�$?+A�F�&?nK���LQ?�lDZrS?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             D�@              �?              �?              �?               @      �?              �?       @       @              �?      �?       @       @       @      �?      �?       @      �?      @      @       @      @      @      @      @      @      @       @      @      "@      @      $@      &@      &@      (@      *@      .@      0@      1@      5@      3@      9@      :@      <@      >@     �A@      C@      C@     �H@     �I@      K@     �O@     @P@     @R@     �S@     �S@      V@      X@     �W@     �P@     �J@      C@      D@     �F@     �C@     �B@      =@      <@      0@      @        
�
layer2/weights/layer2/weights*�	   `����   ����?      $@!   ܹ�)�FRa�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A��/�*>�?�g���w�?yD$��?�QK|:�?����?_&A�o��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   �޴�?   �޴�?      �?!   �޴�?) Jh���d?2�g���w�?���g��?�������:              �?        

	loss/loss��e;��q
      k{f;	R�I�'�A�*�
�
layer1/weights/layer1/weights*�	   �z��   ����?      $@!   �۫@)|m��X�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   @�(ѿ   @9�?      $@!   �J0�?)��QGk��?2�_&A�o�ҿ���ѿ�Z�_��ο�#�h/�?���&�?�/�*>�?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?8/�C�ַ?%g�cE9�?��(!�ؼ?�?>8s2�?yD$��?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?        
�

layer1/layer1/outputs*�
    jw�?     p�@! @�{��@)U���<�@2�        �-���q=+A�F�&?I�I�)�(?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             D�@              �?              �?      �?              �?              �?              �?              �?      �?       @              �?      @      �?              �?       @      �?      @      �?      @      @       @      �?      @      @      @      @      @      @      @      $@      @      "@      $@      &@      (@      ,@      .@      ,@      2@      5@      4@      7@      ;@      ;@      @@      A@     �B@      F@      F@     �I@      K@      O@     �P@     @R@     @S@     �S@      V@      X@      X@     �P@     �J@      C@      D@     �F@     �C@     �B@      =@      <@      1@      @        
�
layer2/weights/layer2/weights*�	   `7���   @C��?      $@!   �n��)�j,fJ�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A��g���w�?���g��?yD$��?�QK|:�?����?_&A�o��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   �3�?   �3�?      �?!   �3�?) .��g?2���g��?I���?�������:              �?        

	loss/loss];�1��q
      k{f;	�g�I�'�A�*�
�
layer1/weights/layer1/weights*�	    ���   �k��?      $@!   ���@)~3�<9�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   �5�ѿ   `qK�?      $@!   8���?)�SY����?2�_&A�o�ҿ���ѿ�Rc�ݒ?^�S���?�/�*>�?�g���w�?���g��?��]$A�?�{ �ǳ�?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�?>8s2�?yD$��?_&A�o��?�Ca�G��?�������:�               @              �?              �?      �?              �?              �?              �?              �?              �?        
�

layer1/layer1/outputs*�
   �Uu�?     p�@! �FǼ�@)=�y��H�@2�        �-���q=+A�F�&?I�I�)�(?a�$��{E?
����G?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             D�@              �?              �?              �?      �?              �?      �?              �?      �?       @              �?      �?      �?      �?      �?      �?       @      �?       @       @      @      @      @      �?      @      @      @      @      @      @      @      @      "@       @      "@      $@      (@      $@      ,@      0@      ,@      2@      4@      6@      5@      :@      =@      A@      ?@      C@     �E@      G@     �I@     �J@     �O@     �P@      R@     �R@     �S@     �U@      X@     @X@     �P@      K@      C@     �D@      F@     �C@     �B@      =@      <@      2@      @        
�
layer2/weights/layer2/weights*�	   `����   ���?      $@!   ��)�W6�ś*@2�ܔ�.�u��S�Fi���yL������������2g�G�A����g��?I���?yD$��?�QK|:�?����?_&A�o��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   ���?   ���?      �?!   ���?) ��N�Wj?2���g��?I���?�������:              �?        

	loss/loss��U;9u�zq
      k{f;	�w�I�'�A�*�
�
layer1/weights/layer1/weights*�	   `@�   ����?      $@!   `'�@)��Yx#�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   ���ҿ   ��]�?      $@!   ��\�?)�ՖҾ�?2�_&A�o�ҿ���ѿ�"�uԖ?}Y�4j�?�/�*>�?�g���w�?���g��?����iH�?��]$A�?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�?>8s2�?yD$��?_&A�o��?�Ca�G��?�������:�               @              �?              �?      �?              �?              �?              �?              �?              �?        
�

layer1/layer1/outputs*�
   `t�?     p�@! `w)��@)�3d��N�@2�        �-���q=>h�'�?x?�x�?U�4@@�$?+A�F�&?a�$��{E?
����G?IcD���L?k�1^�sO?�lDZrS?<DKc��T?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             D�@              �?              �?               @              �?              �?              �?      �?              �?              @      �?       @              @              @       @       @      �?      @      @       @      @      @       @      @      @      @       @      @       @       @      "@      $@      &@      *@      *@      ,@      .@      3@      2@      8@      5@      :@      =@      ?@      B@      C@      E@     �F@      H@     �L@     �O@     �P@     �Q@     @R@     �S@     �U@      X@     @X@     �P@      K@      B@     �D@      F@      D@      B@      >@      ;@      3@      @        
�
layer2/weights/layer2/weights*�	   �Y��   �l��?      $@!   �=5�)
����*@2�ܔ�.�u��S�Fi���yL������������2g�G�A�I���?����iH�?yD$��?�QK|:�?_&A�o��?�Ca�G��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   ��d�?   ��d�?      �?!   ��d�?) ��� �l?2I���?����iH�?�������:              �?        

	loss/lossݶO;���Q
      kuT�	7� J�'�A�*�
�
layer1/weights/layer1/weights*�	   ��   @ƣ�?      $@!   \�@)NR3Ē"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8俰1%���Z%�޿_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   ��Zӿ    �c�?      $@!   H.��?)��T���?2��Ca�G�Կ_&A�o�ҿ���ѿ}Y�4j�?��<�A��?�/�*>�?�g���w�?���g��?����iH�?��]$A�?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?yD$��?�QK|:�?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?        
�

layer1/layer1/outputs*�
    �t�?     p�@! ������@)�5p�P�@2�        �-���q=a�Ϭ(�>8K�ߝ�>>h�'�?x?�x�?IcD���L?k�1^�sO?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             P�@              �?              �?              �?               @      �?      �?               @       @      �?      �?              @              �?      @      @       @      �?      @      @      @      @      @       @       @       @       @      @      @       @      "@      $@       @      &@      (@      *@      ,@      1@      2@      4@      5@      7@      ;@      ;@      @@     �@@      D@      E@     �F@      I@     �L@     �M@     �P@     @Q@     @R@      T@      V@      X@     @X@     @P@     �J@     �B@     �D@      F@     �C@      B@      ?@      ;@      3@      @        
�
layer2/weights/layer2/weights*�	   ����   @e��?      $@!   x�޿)tp7ǳ�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迵���iH�?��]$A�?�QK|:�?�@�"��?_&A�o��?�Ca�G��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	    ��?    ��?      �?!    ��?) ҔI�to?2I���?����iH�?�������:              �?        

	loss/loss��I;���
      Uɗ�	ΡJ�'�A�*�
�
layer1/weights/layer1/weights*�	    �   �v��?      $@!   �K{@)>/���"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8���Z%�޿W�i�bۿ_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   �s�ӿ   `�l�?      $@!   �	�?)NI�J�V�?2��Ca�G�Կ_&A�o�ҿ��<�A��?�v��ab�?�/�*>�?�g���w�?���g��?����iH�?��]$A�?8/�C�ַ?%g�cE9�?!�����?Ӗ8��s�?yD$��?�QK|:�?_&A�o��?�Ca�G��?�������:�               @              �?              �?      �?              �?              �?              �?              �?              �?        
�

layer1/layer1/outputs*�
   `�u�?     p�@! ��@)�f�C�Q�@2�        �-���q=�h���`�>�ߊ4F��>�vV�R9?��ڋ?
����G?�qU���I?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             P�@              �?              �?              �?              �?              �?      �?      �?      �?       @               @       @              �?              �?      �?       @      @      �?       @      @      @      @       @      @      @      @      @      @      @      @      @      $@      @      $@      &@      $@      *@      *@      .@      0@      2@      3@      7@      7@      8@      >@      ?@      A@      C@     �D@     �G@      I@      K@     �O@     �P@     @Q@     @R@     �S@      V@     @X@     @X@      P@     �J@     �B@      D@      F@      D@      B@      >@      ;@      4@      @        
�
layer2/weights/layer2/weights*�	   ����    ���?      $@!    �qݿ)��V���*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迵���iH�?��]$A�?�QK|:�?�@�"��?_&A�o��?�Ca�G��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   ��\�?   ��\�?      �?!   ��\�?) ��"��p?2����iH�?��]$A�?�������:              �?        

	loss/loss��E;B1��A
      Wm;�	�dJ�'�A�*�
�
layer1/weights/layer1/weights*�	    O�   ���?      $@!   Hi@)�+��V�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8���Z%�޿W�i�bۿ_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   ��qԿ   ��l�?      $@!   B���?)<M�"�	�?2��Ca�G�Կ_&A�o�ҿ��<�A��?�v��ab�?�/�*>�?�g���w�?���g��?I���?����iH�?8/�C�ַ?%g�cE9�?!�����?Ӗ8��s�?yD$��?�QK|:�?_&A�o��?�Ca�G��?�������:�               @              �?              �?      �?              �?              �?              �?              �?              �?        
�

layer1/layer1/outputs*�
    �v�?     p�@!  �����@)F7:��Q�@2�        �-���q=pz�w�7�>I��P=�>6�]��?����?+A�F�&?I�I�)�(?���#@?�!�A?�l�P�`?���%��b?5Ucv0ed?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             P�@              �?              �?              �?              �?              @      @              @       @              �?      �?      �?       @       @       @      �?      @      @      @      �?      @       @      @      @      @      @       @      @      @      &@       @      $@      "@      ,@      ,@      *@      1@      3@      1@      8@      8@      :@      ;@      @@      A@      C@      E@     �F@      J@      L@      N@     �P@      Q@      R@      T@     �U@     @X@      X@     @P@     �I@      C@     �D@      F@      C@      B@      ?@      ;@      4@      @        
�
layer2/weights/layer2/weights*�	   ����   �@��?      $@!   �E?ܿ)�͆�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迮�]$A�?�{ �ǳ�?�QK|:�?�@�"��?_&A�o��?�Ca�G��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   ���?   ���?      �?!   ���?) �<�0�q?2����iH�?��]$A�?�������:              �?        

	loss/losse�A;!�.a
      W�.	�J�'�A�*�
�
layer1/weights/layer1/weights*�	    ��   ����?      $@!   ��V@)f�Cv��"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8���Z%�޿W�i�bۿ_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   ���Կ   @�i�?      $@!    =��?)���̄��?2���7�ֿ�Ca�G�Կ_&A�o�ҿ�v��ab�?�/��?�/�*>�?�g���w�?���g��?I���?����iH�?� l(��?8/�C�ַ?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?        
�

layer1/layer1/outputs*�
    �w�?     p�@! @�Hۥ�@)�\�EQ�@2�        �-���q=�7Kaa+?��VlQ.?<DKc��T?ܗ�SsW?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             `�@              �?              �?              �?       @              �?      �?              �?               @               @              �?       @       @      @              �?       @      @      @      @      @       @      @      @      @      @      @       @      @      "@      "@      &@      "@      (@      *@      .@      0@      2@      4@      6@      7@      ;@      :@      @@     �A@      C@      F@      G@     �H@      M@      M@     �P@      Q@      R@      T@     @V@     @X@     �W@      P@      I@     �B@      E@      F@      C@      B@      =@      <@      5@      @        
�
layer2/weights/layer2/weights*�	   `S��   ���?      $@!   0�ۿ)�âh�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迮�]$A�?�{ �ǳ�?�QK|:�?�@�"��?_&A�o��?�Ca�G��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   �p�?   �p�?      �?!   �p�?)@(B�s?2����iH�?��]$A�?�������:              �?        

	loss/loss�>;8B2t�
      i���	��J�'�A�*�
�
layer1/weights/layer1/weights*�	   `#�   �6��?      $@!   �E@)`��݉"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8���Z%�޿W�i�bۿ_&A�o��?�Ca�G��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	    �\տ   �Pm�?      $@!   ��t�?)Pf8�D�?2���7�ֿ�Ca�G�Կ_&A�o�ҿ�v��ab�?�/��?�/�*>�?�g���w�?���g��?I���?����iH�?� l(��?8/�C�ַ?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?        
�
layer1/layer1/outputs*�
   �0x�?     p�@! @�¢��@)�Lv��Q�@2�        �-���q=�uE����>�f����>�S�F !?�[^:��"?�7Kaa+?��VlQ.?uܬ�@8?��%>��:?�T���C?a�$��{E?�lDZrS?<DKc��T?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             X�@              �?              �?              �?              �?              �?              �?              �?              �?               @               @               @      �?      @              @       @      �?      �?      @      @      @       @      @      @      @      @      @      @       @       @      @      $@      "@      (@      *@      (@      ,@      0@      3@      3@      6@      8@      ;@      :@     �@@      A@     �C@      E@      G@      I@      L@      N@     �P@     �P@      R@     @T@     �U@     �X@     �W@     @P@      I@     �B@      D@      F@      C@     �B@      =@      <@      5@      @        
�
layer2/weights/layer2/weights*�	   ����   `��?      $@!   @�ڿ)�4)��*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迮�]$A�?�{ �ǳ�?�QK|:�?�@�"��?_&A�o��?�Ca�G��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   �ϱ?   �ϱ?      �?!   �ϱ?)@�O	g�s?2����iH�?��]$A�?�������:              �?        

	loss/loss8~;;[�i\a
      W�.	,�J�'�A�*�
�
layer1/weights/layer1/weights*�	   ��'�   �q��?      $@!    Z6@)|�6:�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8���Z%�޿W�i�bۿ�Ca�G��?��7��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   `$�տ   �7w�?      $@!   ���?)8|iY��?2���7�ֿ�Ca�G�Կ_&A�o�ҿ�v��ab�?�/��?�/�*>�?�g���w�?���g��?I���?� l(��?8/�C�ַ?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?        
�

layer1/layer1/outputs*�
   �gx�?     p�@! ��˜��@)�I�ظR�@2�        �-���q=})�l a�>pz�w�7�>��VlQ.?��bȬ�0?��%�V6?uܬ�@8?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             d�@              �?              �?              �?              �?      �?              �?               @               @      �?       @      �?       @      �?      �?       @      �?      @       @      @              @      @      @      @      @      @      @      @       @      @      @      (@       @      (@      ,@      (@      .@      .@      2@      5@      6@      6@      <@      ;@      ?@     �A@      C@     �E@      G@     �I@      L@     �N@     �O@     �P@     @R@     @T@     @V@      X@     �W@     @P@      I@     �B@      D@      F@      C@      B@      >@      ;@      6@      @        
�
layer2/weights/layer2/weights*�	    w��   ����?      $@!   x�ٿ)�fQ?�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迦{ �ǳ�?� l(��?�QK|:�?�@�"��?_&A�o��?�Ca�G��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	    U+�?    U+�?      �?!    U+�?)@�P���t?2��]$A�?�{ �ǳ�?�������:              �?        

	loss/loss�Y9;4��Xa
      W�.	dMJ�'�A�*�
�
layer1/weights/layer1/weights*�	   �+�   `~��?      $@!   ��'@)nU�!�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8���Z%�޿W�i�bۿ�Ca�G��?��7��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	   @.7ֿ    �?      $@!   ����?)3ES�!.�?2���7�ֿ�Ca�G�Կ_&A�o�ҿ�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?� l(��?8/�C�ַ?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?        
�

layer1/layer1/outputs*�
    ex�?     p�@!  �Tݜ�@)�z�uT�@2�        �-���q=����?f�ʜ�7
?+A�F�&?I�I�)�(?ܗ�SsW?��bB�SY?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             h�@              �?              �?               @              �?              �?      �?      �?              �?       @      �?              �?      �?      @      �?      @               @      @      @      �?      @      @      @      @      @      @      @       @      @      @       @      "@      "@      (@      *@      *@      .@      1@      2@      5@      4@      5@      <@      <@      @@      A@     �B@      F@     �F@      J@      L@      O@      N@      Q@      R@     @T@     �V@     �W@     �W@     �P@     �H@     �B@      D@      F@      C@      B@      =@      <@      6@      @        
�
layer2/weights/layer2/weights*�	   �D��    ��?      $@!    S]ؿ)[y�04�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迦{ �ǳ�?� l(��?�@�"��?�K?�?�Ca�G��?��7��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   @Cx�?   @Cx�?      �?!   @Cx�?) ��ARu?2��]$A�?�{ �ǳ�?�������:              �?        

	loss/loss��7;=�ߑ1
      j�
�	� J�'�A�*�
�
layer1/weights/layer1/weights*�	    G/�   `��?      $@!   �T@)�Ҋ��"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8���Z%�޿W�i�bۿ�Ca�G��?��7��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	    w�ֿ    s��?      $@!   |zm�?)�dt����?2���7�ֿ�Ca�G�Կ_&A�o�ҿ�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?� l(��?8/�C�ַ?�?>8s2�?yD$��?�QK|:�?_&A�o��?�Ca�G��?�������:�              �?      �?              �?              �?      �?      �?              �?              �?      �?              �?        
�

layer1/layer1/outputs*�
   @Ex�?     p�@!  �n���@)���U�U�@2�        �-���q=>h�'�?x?�x�?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�l�P�`?���%��b?5Ucv0ed?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             p�@              �?              �?              �?              �?      �?              �?      �?       @       @      �?               @      �?      �?      @      �?      @       @      @      �?      @      @      @      @      @      @      @      @      @      @       @      &@      "@      &@      &@      .@      .@      1@      1@      4@      6@      4@      ;@      >@     �@@      ?@     �C@      F@      F@     �I@      L@     �N@      O@     �P@      R@     @T@     �V@      X@      W@      Q@     �H@     �B@      D@     �F@     �B@      B@      =@      <@      6@      @        
�
layer2/weights/layer2/weights*�	    T��    ٤�?      $@!   (F�׿)��n6H�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A迦{ �ǳ�?� l(��?�@�"��?�K?�?�Ca�G��?��7��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	    ���?    ���?      �?!    ���?)@�=��u?2��]$A�?�{ �ǳ�?�������:              �?        

	loss/lossC[6;B��D�
      U���	��J�'�A�*�
�
layer1/weights/layer1/weights*�	   ��2�   @���?      $@!   Ċ
@)���@�"@2��P�1���cI���+�;$��iZ��uo�p�+Se*8���Z%�޿W�i�bۿ�Ca�G��?��7��?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?        
�
layer1/biases/layer1/biases*�	    ��ֿ    ��?      $@!   ��_�?)��6I���?2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?� l(��?8/�C�ַ?�?>8s2�?yD$��?�QK|:�?�Ca�G��?��7��?�������:�              �?              �?              �?              �?      �?      �?              �?              �?      �?              �?        
�
layer1/layer1/outputs*�
   ��w�?     p�@! ���~��@) �_6H\�@2�        �-���q=�iD*L��>E��a�W�>1��a˲?6�]��?�[^:��"?U�4@@�$?�u�w74?��%�V6?�qU���I?IcD���L?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�             d�@              �?              �?              �?              �?              �?               @      �?              @      �?              �?      �?      �?              �?      �?               @      @       @       @      @       @      @       @      @      @      @       @      @      @      @       @      @      $@       @      $@      (@      &@      *@      0@      0@      0@      5@      6@      8@      9@      >@      ?@     �@@     �C@      E@      G@     �H@     �L@      O@      N@     @Q@     �Q@      T@     @V@     �W@      X@     �P@     �H@      B@      D@     �F@     �B@      B@      ?@      :@      7@      @        
�
layer2/weights/layer2/weights*�	   ����   �Y��?      $@!   (W
׿)���q�*@2�ܔ�.�u��S�Fi���yL������������2g�G�A�� l(��?8/�C�ַ?�@�"��?�K?�?�Ca�G��?��7��?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?        
�
layer2/biases/layer2/biases*a	   � ò?   � ò?      �?!   � ò?)@���U v?2��]$A�?�{ �ǳ�?�������:              �?        

	loss/loss�5;� �