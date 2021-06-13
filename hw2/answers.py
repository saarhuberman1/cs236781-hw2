r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

<br>
1. the jacobian matrix is calculated by the all the partial derivatives of the output in respect to the input.
the input dimension is 1024, number of samples is 128 and output dimension is 2048, 
than the entire input is a tensor of shape (128, 1024) and the entire output is a tensor of shape (128, 2048).
the jacobian shape is $(128, 2048, 128, 1024)$.
when we thought of the shape it was more logical for us that the calculation is applied for each instance separatly 
and then the shape will be $(128, 2048, 1024)$. since in former way we mix derivatives between instances and most of 
the derivatives are irrelevant, but when we check in pytorch we got the result we mention before  

2. we represent each cell in our tensor in a single precision point of 4 bytes. than the the amount of memory needed in the RAM or GPU::

$$\begin{align}
\frac{(4 \cdot 128\cdot 2048 \cdot 128 \cdot 1024)} {(1024^3)} = 128 \quad gigabytes
\end{align}$$

"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.01
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr_vanilla = 0.03
    reg = 0.01
    lr_momentum = 3e-3
    lr_rmsprop = 15e-5

    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. the graphs of the dropout vs no-dropout  match what we expected to see. 
The training loss of the no-dropout model decrease much faster, and the accuracy is much higher.
this make sense because the dropout layer randomly drop part of the neurons which make it harder to the network to fit the data. 
in the test set the dropout models performed better than the no-dropout model. we can see that at first the 
no-dropout model test loss descend very first and than stop descending while the train accuracy continue to descend. 
as we learn this model started to over fit the training data. 
the dropout models continue to improve on the test set and recieve better results since the dropout layer make it harder 
for the model to over fit the training data and help generalized the model. 


2. when looking on the performance on the training set, the low dropout perform much better. the accuracy rise much faster 
and to significant higher result. as in section 1 this can be explained by that the high percentage of randomly drop out 
neurons are making it much  harder for the model to fit the data.
when looking on the test set, the high dropout learn much slower, and got lower accuracy, the significant 0.8 dropout 
make it too hard for the model to learn. 


"""

part2_q2 = r"""
**Your answer:**

Yes, this is possible. the cross entropy loss function is $-\sum_{i}y_i log(\hat{y_i})$, 
it looks at the predicted distribution that are coming from the softmax and return the $-log(\hat{y_i})$ value of 
the true label, meaning this loos is not only define by if the max probability label is equal to the correct label as in 
accuracy, but also affected by how much the classifier is sure of its decision.

if the model will predict correctly more instances in the test set, but the output distribution of the softmax will be 
more wide and close to uniform the test loss will increase and also the accuracy will increase.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1) Number of parameters of a conv layer = $ (width of conv kernel \times height of conv kernel \times number of filters in the previous layer+1)\times number of filters in current layer)$ 
Number of parameters for regular ResBlock: 

both conv_layer1 and conv_layer2 have $64*(64*3^2+1)=36928$ parameters

So in total we have $36928 \times 2 = 73,856$ parameters


Number of parameters of bottleneck ResBlock:

Layer1:  $64\times(256 \times 1^2+1)=16,448$

Layer2: $64\times(64 \times 3^2+1)=36928$

Layer3: $256 \times (64 \times 1^2 +1) = 16640$

So in total we have $36928 + 16,448 +16,640 = 70,016$ 


2)	Number of FLOPs of a conv layer: $ 2HW\times(C_{in} k_w k_h +1)C_{out}$ where $H$ is the 
layer input hight and $W$ is the layer input width. In our case these stay constant through the
 ResBlocks as padding is used. As we can see, the number of FLOPs in each conv layer is the 
 number of parameters of each layer multiplied by $2HW$
 
Number of parameters for regular ResBlock:
In total we have $2HW*73,856$ FLOPs

Number of parameters of bottleneck ResBlock:
In total we have $2HW*70,016$ FLOPs

So in terms of FLOPs, the regular ResBlock performs about $2HW*3000$ more FLOPs than the 
bottleneck ResBlock.

3)	Comparing the ability to combine the input:
1. Within feature maps: regular ResBlock has a higher ability to combine the input within 
a feature map. In the regular ResBlock case, each output feature of a single layer depends on
 $3*3=9$ input features, so overall one output feature depends on at least $5*5=25$ features 
 (in the case of stride=1). While in the case of the bottleneck ResBlock, an output feature of 
 the first and last layers depend only on 1 input feature and the second layer on $3*3=9$input 
 features so overall one output feature depends on 9 input features. 

2. Across feature maps:   due to it’s structure, the bottleneck ResBlock allows to reach a compact
 representation of the feature map as it projects the input feature map into smaller channel size 
 then projects it back to the original. While regular ResBlocks act as regular conv layers 
 (no special structure is assumed) in terms of feature map projections.


"""

part3_q2 = r"""
1) A deeper network is expected to perform better (reach higher accuracy), 
up to a certain number of layers where the network will become “too deep” and then 
we’d see the effect of the vanishing gradient problem and the network won’t be able to learn 
anymore. As we can observe from the plots, the best accuracy was reached when L=4 (with K=64 
the difference between the test accuracies when L=2 and L=4 is noticeable). This while networks 
with depth 8 and 16 (for both K=32 and K=64) could not learn (they result in accuracy of about 
10% with is equivalent to random guessing in this case as we have 10 possible classes).


2) As we mentioned in part 1 of this question, for L=8 and L=16 the network wasn’t 
trainable. The reason for this is the vanishing gradients problem. Two possible solutions 
are (1) using residual networks and (2) using batch normalization.


"""

part3_q3 = r"""
In exp 2 we see consistent results with exp 1.1. For small number of layers 
($L=2$ and $L=4$) the network was able to learn and reached accuracy of $55-70%$ 
(varying with the number of filters per layer). We observe that for $L=2$ the best 
accuracy is reached with smaller number of filters per layer (K=32 and K=64 performed 
better than K=128 and K=256). Where for L=4 we observer that the more filters used 
the higher the accuracy that was reached. For L=8 the network was not trainable. 
This can be caused by the vanishing gradients problem (like what we saw in 1.1)
 or can be caused because of the large parameter space that needs to be toned.

"""

part3_q4 = r"""
This experiment uses a cnn with different number of filters per layer.
 We can see that for $L=1$ and $L=2$ (networks with depth 3 and 6),
  the network reached the highest accuracy we got so far comparing to exp 1.1 
  and 1.2 (about 70%). Where for $L=3$ and $L=4$ $ (depth 9 and 12), the network was 
  not trainable. Once more, we think this is due to the vanishing gradient problem as
   we explained in the previous questions. 

"""

part3_q5 = r"""
In this experiment we trained a ResNet 
(In contrast to all previous experiments where we trained a CNN). 

In exp 1.1 we saw that the CNN was not trainable with $L\geq 8$ and K=32 
while in this experiment we can see that the network reached  ~75% accuracy thanks 
to the skip connections in the residual blocks that help with solving the vanishing 
gradients problem. 
In exp 1.3 we saw that with $L\geq 3$ and $K=[64-128-256]$ the network was not trainable 
while in this exp we can see that the ResNet is reaching a reasonable accuracy and not facing 
the same problem even with deeper network architectures. 


"""

part3_q6 = r"""
In this part we chose to build a network that consists on convolutional layers while using 
batch normalization and dropout and added a bottleneck ResBlock in the middle on the network. 
We use batch normalization to reduce the effect of vanishing gradients and make the learning more
 stable. We use dropout to prevent the model from overfitting and we use the bottleneck ResBlock
  to get a more compact representation of the parameters to try to train deeper models.
We can see that the network with $L=3$ could reach about 80% accuracy (the highest among all other 
experiments). 
We can see that for L=12 the network is over-fitting. This might be due to the large number of 
parameters and increasing the dropout might help to solve this, however we did not try it.

Comparing the results to exp 1, we can see that using the ResBlock and batch normalization allowed 
the network to learn regardless of the deep architecture. 

"""