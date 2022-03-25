**Self-Supervised Learning**，又称为自监督学习，一般机器学习分为有监督学习，无监督学习和强化学习。 而 Self-Supervised Learning 是无监督学习里面的一种，主要是希望能够学习到一种**通用的特征表达**用于**下游任务 (Downstream Tasks)**。 其主要的方式就是通过自己监督自己。首先是 kaiming 的 MoCo 引发一波热议， Yann Lecun也在 AAAI 上讲 Self-Supervised Learning 是未来的大势所趋。

总结下 Self-Supervised Learning 的方法，用 4 个英文单词概括一下就是：

> **Unsupervised Pre-train, Supervised Fine-tune.**

在预训练阶段我们使用**无标签的数据集 (unlabeled data)**，因为有标签的数据集**很贵**，打标签得要多少人工劳力去标注，那成本是相当高的，所以这玩意太贵。相反，无标签的数据集网上随便到处爬，它**便宜**。在训练模型参数的时候，我们不追求把这个参数用带标签数据从**初始化的一张白纸**给一步训练到位，原因就是数据集太贵。于是 **Self-Supervised Learning** 就想先把参数从 **一张白纸** 训练到 **初步成型**，再从 **初步成型** 训练到 **完全成型**。注意这是2个阶段。这个**训练到初步成型的东西**，我们把它叫做 **Visual Representation**。预训练模型的时候，就是模型参数从 **一张白纸** 到 **初步成型** 的这个过程，还是用无标签数据集。等我把模型参数训练个八九不离十，这时候再根据 **下游任务 (Downstream Tasks)** 的不同去用带标签的数据集把参数训练到 **完全成型**，那这时用的数据集量就不用太多了，因为参数经过了第1阶段就已经训练得差不多了。

第1个阶段不涉及任何下游任务，就是拿着一堆无标签的数据去预训练，没有特定的任务，这个话用官方语言表达叫做：**in a task-agnostic way**。第2个阶段涉及下游任务，就是拿着一堆带标签的数据去在下游任务上 Fine-tune，这个话用官方语言表达叫做：**in a task-specific way** 。

**以上这些就是 Self-Supervised Learning 的核心思想**，如下图所示。

![self sup](https://user-images.githubusercontent.com/50043212/160054981-274b2201-ffaf-4c95-adca-2d3f291a94ea.jpg)


- **SimCLR 原理分析**

> **论文名称：A Simple Framework for Contrastive Learning of Visual Representations**

**SimCLR** 是Hinton的Google Brain团队在 **Self-Supervised Learning** 领域的一个系列经典工作。先来通过图2直观地感受下它的性能：SimCLR (4×) 这个模型可以在 ImageNet 上面达到 **76.5%** 的 **Top 1** Accuracy，**比当时的 SOTA 模型高了7个点**。如果把这个预训练模型用 **1%的ImageNet的标签**给 **Fine-tune** 一下，借助这一点点的有监督信息，SimCLR 就可以再达到 **85.5%** 的 **Top 5** Accuracy。

![85%](https://user-images.githubusercontent.com/50043212/160055112-e6b4156f-77ed-41df-bf14-ee7e1943017b.jpg)

**Self-Supervised Learning 的目的一般是使用大量的无 label 的资料去Pre-train一个模型，这么做的原因是无 label 的资料获取比较容易，且数量一般相当庞大，我们希望先用这些廉价的资料获得一个预训练的模型，接着根据下游任务的不同在不同的有 label 数据集上进行 Fine-tune 即可**。

作为 Self-Supervised Learning 的工作之一，SimCLR 自然也遵循这样的思想。其一个核心的词汇叫做：**Contrastive**。



这个词翻译成中文是 **对比** 的意思，它的实质就是：**试图教机器区分相似和不相似的事物**。

![4](https://user-images.githubusercontent.com/50043212/160055135-b8618937-ed9e-4a4d-9bcb-85f31ded42c9.jpg)

这个话是什么意思呢？比如说现在我们有任意的 4 张 images，如下图所示。前两张都是dog 这个类别，后两张是其他类别，以第1张图为例，我们就希望**它与第2张图的相似性越高越好，而与第3，第4张图的相似性越低越好**。

但是以上做法**其实都是很理想的情形**，因为：

1. 我们只有大堆images，没有任何标签，不知道哪些是 dog 这个类的，哪些是其他类的。
2. 没办法找出哪些图片应该去 Maximize Similarity，哪些应该去 Minimize Similarity。
![5](https://user-images.githubusercontent.com/50043212/160055574-2b128486-a724-451a-8a95-eb75950ad6ac.jpg)

所以，SimCLR是怎么解决这个问题的呢？它的framework如下图所示：

假设现在有**1张**任意的图片 ![[公式]](https://www.zhihu.com/equation?tex=x) ，叫做Original Image，先对它做数据增强，得到2张增强以后的图片 ![[公式]](https://www.zhihu.com/equation?tex=x_i%2Cx_j) 。注意数据增强的方式有以下3种：

- 随机裁剪之后再resize成原来的大小 (Random cropping followed by resize back to the original size)。
- 随机色彩失真 (Random color distortions)。
- 随机高斯模糊 (Random Gaussian Deblur)。

接下来把增强后的图片 ![[公式]](https://www.zhihu.com/equation?tex=x_i%2Cx_j) 输入到Encoder里面，注意这2个Encoder是共享参数的，得到representation ![[公式]](https://www.zhihu.com/equation?tex=h_i%2Ch_j) ，再把 ![[公式]](https://www.zhihu.com/equation?tex=h_i%2Ch_j) 继续通过 Projection head 得到 representation ![[公式]](https://www.zhihu.com/equation?tex=z_i%2Cz_j) ，这里的2个 Projection head 依旧是共享参数的，且其具体的结构表达式是：

![[公式]](https://www.zhihu.com/equation?tex=z_i%3Dg%28h_i%29%3DW%5E%7B%282%29%7D%5Csigma+%28W%5E%7B%281%29%7Dh_i%29%5C%5C)

接下来的目标就是最大化同一张图片得到的 ![[公式]](https://www.zhihu.com/equation?tex=z_i%2Cz_j) 。
![框架](https://user-images.githubusercontent.com/50043212/160055616-cc435788-7887-43ff-a88d-ca9450b01e69.jpg)

以上是对SinCLR框架的较为笼统的叙述，下面具体地看每一步的做法：

回到起点，一开始我们有的training corpus就是一大堆 unlabeled images，如下图所示。

![6](https://user-images.githubusercontent.com/50043212/160055645-31adaa95-60c7-4301-8600-1966de75ccd6.jpg)

- **1.1 数据增强**

比如batch size的大小是 ![[公式]](https://www.zhihu.com/equation?tex=N) ，实际使用的batch size是8192，为了方便我们假设 ![[公式]](https://www.zhihu.com/equation?tex=N%3D2) 。
![7](https://user-images.githubusercontent.com/50043212/160055680-2ce9ef60-2e00-4d0c-b59d-0cc90c8738e9.jpg)

注意数据增强的方式有以下3种：

- 随机裁剪之后再resize成原来的大小 (Random cropping followed by resize back to the original size)。代码：

```python3
torchvision:transforms:RandomResizedCrop
```

- 随机色彩失真 (Random color distortions)。代码：

```python3
from torchvision import transforms
def get_color_distortion(s=1.0):
# s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
    rnd_color_jitter,
    rnd_gray])

    return color_distort
```

- 随机高斯模糊 (Random Gaussian Deblur)。

```python3
random (crop + flip + color jitter + grayscale)
```

![8](https://user-images.githubusercontent.com/50043212/160055707-ae341b12-d3fb-44a0-8961-d05f1ee7b1c9.jpg)

对每张图片我们得到2个不同的数据增强结果，所以1个Batch 一共有 4 个 Image。
![9](https://user-images.githubusercontent.com/50043212/160055720-769b6071-5569-4777-9cdc-ffe421ae2119.jpg)

- **1.2 通过Encoder获取图片表征**

第一步得到的2张图片 ![[公式]](https://www.zhihu.com/equation?tex=x_i%2Cx_j) 会通过Encoder获取图片的表征，如下图所示。所用的编码器是通用的，可以用其他架构代替。下面显示的2个编码器共享权重，我们得到向量 ![[公式]](https://www.zhihu.com/equation?tex=h_i%2Ch_j) 。

![10](https://user-images.githubusercontent.com/50043212/160055743-76cb5da5-1cef-4e6b-90b1-dd6c525bf5ed.jpg)

本文使用了 **ResNet-50** 作为 **Encoder**，输出是 **2048** 维的向量 ![[公式]](https://www.zhihu.com/equation?tex=h) 。



- **1.3 预测头**

使用预测头 Projection head。在 SimCLR 中，Encoder 得到的2个 visual representation再通过Prediction head (![[公式]](https://www.zhihu.com/equation?tex=g%28.%29))进一步提特征，预测头是一个 2 层的MLP，将 visual representation 这个 2048 维的向量![[公式]](https://www.zhihu.com/equation?tex=h_i%2Ch_j)进一步映射到 128 维隐空间中，得到新的representation ![[公式]](https://www.zhihu.com/equation?tex=z_i%2Cz_j)。利用 ![[公式]](https://www.zhihu.com/equation?tex=z_i%2Cz_j) 去求loss 完成训练，训练完毕后扔掉预测头，保留 Encoder 用于获取 visual representation。
![11](https://user-images.githubusercontent.com/50043212/160055756-76fd0cdf-d02d-47fd-af2f-004e8f7b8899.jpg)

- **1.4 相似图片输出更接近**

到这一步以后对于每个Batch，我们得到了如下图所示的Representation ![[公式]](https://www.zhihu.com/equation?tex=z_1%2C...%2Cz_4) 。

![12](https://user-images.githubusercontent.com/50043212/160055776-aca6c84b-d95f-4a97-b38d-6deff535434f.jpg)

首先定义Representation之间的相似度：使用余弦相似度Cosine Similarity：
![13](https://user-images.githubusercontent.com/50043212/160055795-c60e7e55-5658-4d53-90db-809adb654b23.jpg)

Cosine Similarity把计算两张 Augmented Images ![[公式]](https://www.zhihu.com/equation?tex=x_i%2Cx_j) 的相似度转化成了计算两个Projected Representation ![[公式]](https://www.zhihu.com/equation?tex=z_i%2Cz_j) 的相似度，定义为：

![[公式]](https://www.zhihu.com/equation?tex=s_%7Bi%2Cj%7D%3D%5Cfrac%7B%5Ccolor%7Bcrimson%7D%7Bz_i%7D%5E%5Ctop+%5Ccolor%7Bcrimson%7D%7Bz_j%7D%7D%7B%5Ctau+%7C%7C%5Ccolor%7Bcrimson%7D%7Bz_i%7D%7C%7C%7C%7C%5Ccolor%7Bcrimson%7D%7Bz_j%7D%7C%7C%7D%5C%5C)

式中， ![[公式]](https://www.zhihu.com/equation?tex=%5Ctau) 是可调节的Temperature 参数。它能够scale 输入并扩展余弦相似度`[-1, 1]`这个范围。

使用上述公式计算batch里面的每个Augmented Images ![[公式]](https://www.zhihu.com/equation?tex=x_i%2Cx_j) 的成对余弦相似度。 如下图所示，在理想情况下，狗的增强图像之间的相似度会很高，而狗和鲸鱼图像之间的相似度会较低。

![14](https://user-images.githubusercontent.com/50043212/160055827-dcd522ef-7fb1-4dff-bd49-1044eaf8df5d.jpg)

现在我们有了衡量相似度的办法，但是这还不够，要最终转化成一个能够优化的 Loss Function 才可以。

SimCLR用了一种叫做 **NT-Xent loss** (**Normalized Temperature-Scaled Cross-Entropy Loss**)的对比学习损失函数。

我们先拿出Batch里面的第1个Pair：

![15](https://user-images.githubusercontent.com/50043212/160055849-3c24cfc0-5137-4c3d-a9d7-ce2c98718978.jpg)

使用 softmax 函数来获得这两个图像相似的概率：

![16](https://user-images.githubusercontent.com/50043212/160055866-f0f42403-83fe-4119-9810-d908c1806e20.jpg)

这种 softmax 计算等价于获得第2张增强的狗的图像与该对中的第1张狗的图像最相似的概率。 在这里，分母中的其余的项都是其他图片的增强之后的图片，也是negative samples。

所以我们希望上面的softmax的结果尽量大，所以损失函数取了softmax的负对数：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7Bl%7D%28i%2Cj%29%3D-%5Clog%5Cfrac%7B%5Cexp%28s_%7Bi%2Cj%7D%29%7D%7B%5Csum_%7Bk%3D1%7D%5E%7B2N%7D%7B%5Cmathbf%7B1%7D%5Bk%21%3Di%5D%5Cexp%28s_%7Bi%2Ck%7D%29%7D%7D%5C%5C)
![17](https://user-images.githubusercontent.com/50043212/160055888-5d4a7baf-a669-423d-bf0c-511e1f96c37f.jpg)

再对同一对图片交换位置以后计算损失：

![18](https://user-images.githubusercontent.com/50043212/160055912-f282e232-092f-4795-aa82-aea6d6e65052.jpg)

最后，计算每个Batch里面的所有Pair的损失之和取平均：

![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Cfrac%7B1%7D%7B2N%7D%5Csum_%7Bk%3D1%7D%5E%7BN%7D%5B%7Bl%282k-1%2C2k%29%7D%2Bl%282k%2C2k-1%29%5D%5C%5C)

![19](https://user-images.githubusercontent.com/50043212/160055934-51a47132-6e56-430d-9d6a-80cae396a2e8.jpg)

- **1.5 对下游任务Fine-tune**

至此我们通过对比学习，巧妙地在没有任何标签的情况下训练好了 SimCLR 模型，使得其Encoder的输出可以像正常有监督训练的模型一样表示图片的Representation信息。所以接下来就是利用这些 Representation的时候了，也就是在下游任务上Fine-tune。一旦 SimCLR 模型在对比学习任务上得到训练，它就可以用于迁移学习，如 ImageNet 分类，如下图所示。此时在下游任务上 Fine-tune 模型时需要labeled data，但是数据量可以很小了。

![20](https://user-images.githubusercontent.com/50043212/160055949-37e3ec80-b9f4-40df-875f-d7daa8bb3794.jpg)

**性能：**

SimCLR (4×) 这个模型可以在 ImageNet 上面达到 **76.5%** 的 **Top 1** Accuracy，**比当时的 SOTA 模型高了7个点**。如果把这个预训练模型用 **1%的ImageNet的标签**给 **Fine-tune** 一下，借助这一点点的有监督信息，SimCLR 就可以再达到 **85.5%** 的 **Top 5** Accuracy。



**FAQ1：这个 76.5% 和 85.5% 是怎么得到的呢？**

**答1：76.5% 是通过Linear Evaluation得到的。**

按照上面的方式进行完Pre-train之后，Encoder部分和Projection head部分的权重也就确定了。那么这个时候我们去掉Projection head的部分，在Encoder输出的 ![[公式]](https://www.zhihu.com/equation?tex=h_i%2Ch_j) 之后再添加一个**线性分类器 (Linear Classifier)**，它其实就是一个FC层。那么我们使用全部的 ImageNet **去训练这个 Linear Classifier**，具体方法是把预训练部分，即 ![[公式]](https://www.zhihu.com/equation?tex=h_i%2Ch_j) 之前的权重frozen住，只训练线性分类器的参数，那么 Test Accuracy 就作为 a proxy for representation quality，就是76.5%。

**85.5% 是通过Fine-tuning得到的。**

按照上面的方式进行完Pre-train之后，Encoder部分和Projection head部分的权重也就确定了。那么这个时候我们去掉Projection head的部分，在Encoder输出的 ![[公式]](https://www.zhihu.com/equation?tex=h_i%2Ch_j) 之后再添加一个**线性分类器 (Linear Classifier)**，它其实就是一个FC层。那么我们使用 **1%的ImageNet 的标签** **去训练整个网络**，不固定 Encoder 的权重了。那么最后的 Test Accuracy 就是85.5%。

Linear Evaluation 和 Fine-tuning的精度的关系如下图所示：当Linear Evaluation 的精度达到76.5% Top1 Accuracy时，Fine-tuning的精度达到了50多，因为Fine-tuning 只使用了1%的标签，而Linear Evaluation 使用了100%的标签。

![21](https://user-images.githubusercontent.com/50043212/160056298-41724ffe-b545-41b0-81c4-9d9b7d38ff05.png)



**FAQ2：Projection head 一定要使用非线性层吗？**

**答2：**不一定。作者尝试了3种不同的 Projection head 的办法，分别是：Non-Linear， Linear 层和 Identity mapping，结果如下图所示。发现还是把Projection head ![[公式]](https://www.zhihu.com/equation?tex=g%28.%29) 设置成非线性层 Non-Linear 比较好。 Non-Linear 比 Linear 层要涨3%的Top 1 Accuracy，比 Identity mapping 层要涨10%的Top 1 Accuracy。

而且，作者的另一个发现是 Projection head 前面的 hidden layer 相比于 Projection head后面的 hidden layer 更好。那这个更好是什么意思呢？

就是假设我们把 Projection head 前面的 hidden layer ![[公式]](https://www.zhihu.com/equation?tex=h) 作为图片的representation的话，那么经过线性分类层得到的模型性能是好的。如果把Projection head 后面的 hidden layer ![[公式]](https://www.zhihu.com/equation?tex=g%28h%29) 作为图片的representation的话，那么经过线性分类层得到的模型性能不好，如下图所示，是对 ![[公式]](https://www.zhihu.com/equation?tex=h) 或者 ![[公式]](https://www.zhihu.com/equation?tex=g%28h%29) 分别训练一个额外的MLP， ![[公式]](https://www.zhihu.com/equation?tex=h) 或者 ![[公式]](https://www.zhihu.com/equation?tex=g%28h%29)的hidden dimension 都是2048。

![22](https://user-images.githubusercontent.com/50043212/160056334-00c508bf-911a-4fa7-a70d-3ce7a0e34f83.png)
![23](https://user-images.githubusercontent.com/50043212/160056358-f8d7ae94-2a53-413e-80a6-a4876607e7d6.jpg)



**FAQ3：NT-Xent loss (Normalized Temperature-Scaled Cross-Entropy Loss)的对比学习损失函数如何代码实现？**

**答3：**

```python3
import tensorflow as tf
import numpy as np

def contrastive_loss(out,out_aug,batch_size=128,hidden_norm=False,temperature=1.0):
    if hidden_norm:
        out=tf.nn.l2_normalize(out,-1)
        out_aug=tf.nn.l2_normalize(out_aug,-1)
    INF = np.inf
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2) #[batch_size,2*batch_size]
    masks = tf.one_hot(tf.range(batch_size), batch_size) #[batch_size,batch_size]
    logits_aa = tf.matmul(out, out, transpose_b=True) / temperature #[batch_size,batch_size]
    logits_bb = tf.matmul(out_aug, out_aug, transpose_b=True) / temperature #[batch_size,batch_size]
    logits_aa = logits_aa - masks * INF # remove the same samples in out
    logits_bb = logits_bb - masks * INF # remove the same samples in out_aug
    logits_ab = tf.matmul(out, out_aug, transpose_b=True) / temperature
    logits_ba = tf.matmul(out_aug, out, transpose_b=True) / temperature
    loss_a = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss=loss_a+loss_b
    return loss,logits_ab

'''
假设batch_size=3, out 和 out_aug 分别代码 原始数据和增强数据的representation
out : [a1,a2,a3] 
out_aug : [b1,b2,b3] 

labels：
[batch_size,2*batch_size] batch_size=3 
1 0 0 0 0 0 
0 1 0 0 0 0 
0 0 1 0 0 0 

mask：
 [batch_size,batch_size]
1 0 0
0 1 0
0 0 1

logits_aa [batch_size,batch_size]
a1*a1, a1*a2, a1*a3 
a2*a1, a2*a2, a2*a3  
a3*a1, a3*a2, a3*a3 

logits_bb [batch_size,batch_size]
b1*b1, b1*b2, b1*b3 
b2*b1, b2*b2, b2*b3 
b3*b1, b3*b2, b3*b3 

logits_aa - INF*mask # delete same samples
-INF,  a1*a2,  a1*a3 
a2*a1, -INF,  a2*a3 
a3*a1, a3*a2,  -INF 

logits_bb - INF*mask  # delete same samples
-INF,  b1*b2, b1*b3 
b2*b1, -INF,  b2*b3 
b3*b1, b3*b2, -INF 

logits_ab [batch_size,batch_size]
a1*b1, a1*b2, a1*b3 
a2*b1, a2*b2, a2*b3 
a3*b1, a3*b2, a3*b3

logtis_ba [batch_size,batch_size]
b1*a1, b1*a2,  b1*a3 
b2*a1, b2*a2, b2*a3
b3*a1, b3*a2, b3*a3

concat[logits_ab,logits_aa]:
a1*b1, a1*b2, a1*b3,  -INF,  a1*a2, a1*a3 
a2*b1, a2*b2, a2*b3, a2*a1, -INF, a2*a3
a3*b1, a3*b2, a3*b3, a3*a1, a3*a2, -INF
only a1*b1, a2*b2, a3*b3  are positives

concat [logits_ab,logits_bb]:
b1*a1, b1*a2,  b1*a3, -INF,  b1*b2, b1*b3 
b2*a1, b2*a2, b2*a3, b2*b1, -INF, b2*b3
b3*a1, b3*a2, b3*a3, b3*b1, b3*b2, -INF
only b1*a1, b2*a2, b3*a3  are positives, so calculate the softmax_cross_entropy with labels

'''
```

简单分析下代码最后会得到：

```python3
a1*b1, a1*b2, a1*b3,  -INF,  a1*a2, a1*a3 
```

对它做softmax，这里的 a1*a2 就代表第1张图片的2个 Augmented Images 的 similarity，a1*b2 代表第1张图片的第1个 Augmented Images 和第2张图片的第2个Augmented Images 的similarity。



**FAQ4：SimCLR 的性能与 Batch size 的大小和训练的长度有关吗？**

**答4：**有关系。如下图所示，作者发现当使用较小的 training epochs 时，大的 Batch size 的性能显著优于小的 Batch size 的性能。作者发现当使用较大的 training epochs 时，大的 Batch size 的性能和小的 Batch size 的性能越来越接近。这一点其实很好理解：在对比学习中，较大的 Batch size 提供更多的 negative examples，能促进收敛。更长的 training epochs 也提供了更多的 negative examples，改善结果。

![24](https://user-images.githubusercontent.com/50043212/160056446-2935a972-44a8-4f1a-8d33-19bc464d3f53.jpg)

