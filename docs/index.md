---
layout: default
---

# Intro
We set out to try a new approach on generalization in machine learning algorithms. It is often hard to train a model on predicting labels for a given set of data that will perform well on new sets of data. Training a new model on new datasets may also be infeasible due difficulties like: a lack of labels or difficulty in collecting new data. With our new approach to generalization, we hope to improve a neural network model's ability to classify to new datasets. To accomplish this, we are using a Variational Autoencoder (VAE) as well as a Generative Adversarial Network (GAN) to achieve accurate image-to-image translation. We use this translation as a form of  domain adaptation. As a proof-of-concept, we explore this model's ability to adapt the lesser known KannadaMNIST dataset to images similar to the more well known MNIST dataset. The resulting images are then classified with a pre-trained MNIST model.

## MNIST and Kannada-MNIST

The MNIST dataset is a commonly used dataset in machine learning. This dataset consists of images of size 28 pixels by 28 pixels. Each image contains a hand-drawn arabic numeral between 0 to 9 inclusive. Typically, this dataset is used to train a supervised machine learning model to predict an arabic numeral label for a given 28 by 28 pixel image.

The Kannada-MNIST, or K-MNIST, dataset is similar to the MNIST dataset except the images are of hand-drawn Kannada numerals instead of arabic numerals. For reference, Kannada is a language predominantly spoken in Karnataka (a state in the southwest of India). This dataset is fairly new, and some are still researching on how to train the most accurate model to predict the labels for this dataset.

The followings are example images of MNIST and K-MNIST data for each numeric class:

| MNIST | Kannada |
|:-|:-|
| <img src="{{ site.baseurl }}/assets/images/MNIST_labeled.png" height="500" /> | <img src="{{ site.baseurl }}/assets/images/KMNIST_labeled.png" height="500" /> |
| Credit: https://www.researchgate.net/figure/Example-images-from-the-MNIST-dataset_fig1_306056875 | Credit: https://towardsdatascience.com/a-new-handwritten-digits-dataset-in-ml-town-kannada-mnist-69df0f2d1456 |

# Background

## Variational Autoencoders
In general, autoencoder is a form of unsupervised learning algorithm that implements the use of neural networks with the typical goal of data compression and dimensionality reduction.

Overall, the structure of an autoencoder can be outlined as followed (1):

<p align="center">
    <img src="assets/images/autoencoders.png" alt="Autoencoders" />
    <br>
    <em>Autoencoder</em>
</p>

* Encoder: the neural network responsible that is responsible for learning how to perform dimensionality reduction and produce a representation  of the reduced data
* Bottleneck (latent space): the representation, in the form of a vector, of the input after compression is performed
* Decoder: the neural network responsible for reproducing the original input from the bottleneck

Essentially, dimensionality reduction is performed through the training of the encoder and decoder in order to tune the neural networks' parameters and minimize reconstruction loss between input and output. While autoencoders have been used and proven to be effective models for data compression, they cannot be used to generate new content just by having the decoder taking a sample vector within the latent space. This stems from the lack of regularization of the latent space by the autoencoder, whose learning and training processes direct towards the single goal of encoding and decoding the input. With the latent space constructed as distinct clusters by the encoder, thus exhibiting discontinuities, random sampling from such latent space and feeding it back into the decoder will result in non-meaningful output.

Variational Autoencoder (VAE) is a specific framework within "generative modeling", which in itself, is an area of machine learning that deals with distribution models of data points within a high dimensional space. While structurally similar to an autoencoder by which it also contains an encoder, decoder and latent space, to accomplish the generative process, VAE's encoder produces a distribution (enforced to approximate a standard normal distribution) within the latent space rather than encoding a vector representation (2).

<p align="center">
    <img src="assets/images/vae.png" alt="VAE" />
    <br>
    <em>Variational Autoencoder</em>
</p>


Under this model, the generation of new information is performed through the sampling within the distribution and processing of the decoder. To analyze the competency of VAE model, rather than implementing the use of reconstruction loss, analysis is typically performed using a combination of generative loss (the difference between the generated image and a real image) and latent loss (the Kullback-Leibler divergence between the latent distribution and unit Gaussian).

Variation autoencoders have been incorporated in literatures and practical scenarios for many different purposes, including the interpolation of facial images with respect to different attributes (age, hair color, expression, etc.). For this particular project, Variational Autoencoders is combined with Generative Adversarial Networks as part of a UNIT framework that is implemented for image-to-image translation, specifically, the translation from Kannada MNIST to MNIST digits.

reference:
(1) https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726
(2) https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73


## Generative Adversarial Networks
GAN stands for Generative Adversarial Network, which are deep-learning based generative models. GANs are a model architecture for training generative models which are widely used to translate inputs from one domain to another. GANs were first introduced in 2014 by Ian Goodfellow et al in a paper titled "[Generative Adversarial Networks]([https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661))" . While initially proposed as a model for unsupervised learning, GANs have also proved to be useful for semi-supervised learning, fully supervised learning and reinforcement learning.

The GAN model involves two sub-models:

1. **Generator Model** - This is a model that is used to generate new examples from the problem domain.

	The input to the model is a vector from a multidimensional space. After training with the dataset, this multidimensional space is mapped to corresponding points in the problem domain. This forms a compressed representation of the multidimensional data space.

	After training, the generator model is used to generate new samples.

2. **Discriminator Model** - This model is used to classify example inputs based on whether they come from the problem domain or from the generated examples.

	The model inputs an example from the domain (real or generated) and classifies it with a binary label *real* or *fake*. The *real* examples come from the training dataset, while the *fake* examples come from the generator model.

	Typically, after training is complete, the discriminator model is discarded since we are more interested in the generator model to generate more samples.

<p align="center">
    <img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/04/Example-of-the-Generative-Adversarial-Network-Model-Architecture.png" alt="GAN" />
    <br>
    <em>Generative Adversarial Network</em>
</p>

A key use of generative adversarial networks comes in image-to-image translation, to map images from the input domain to a different output domain.

# Image-to-Image Translation Networks
In this project, we use a framework that combines VAE and GAN to perform image-to-image translation tasks.
Specifically, we adopt the UNIT framework proposed by [Liu et al. (2017)](#liu2017), which was used in unsupervised image-to-image translation tasks, while we further extend it to semi-supervised and fully-supervised image translation tasks.

## Framework
The overall framework of our proposed model is depicted in the following figure.

<p align="center">
    <img src="{{ site.baseurl }}/assets/images/framework.png" alt="SEMIT" height="480" />
    <br>
    <em>Image-to-image translation networks. </em>
</p>

It is a combination of VAE and GAN architecture that consists of the following modules:

1. **Encoders**: each encoder $$E_i$$ encodes samples from a source domain data $$X_1$$ or a target domain data $$X_2$$ into a shared latent space<sup>[*](#shared)</sup> $$Z$$.

2. **Decoders/Generators**: each decoder (in terms of VAE) or generator (in terms of GAN) $$G_i$$ reconstructs samples $$\widetilde{X}_{i}^{j}$$ using latent vectors $$z$$ where the subscript $$i$$ means the decoder's own domain and the superscript $$j$$ means the sample's origin domain. For example, $$\widetilde{X}_1^2$$ represents the samples reconstructed in $$X_1$$ domain using the latent vectors encoded using the input data from $$X_2$$ domain.

3. **Discriminators**: each discriminator $$D_i$$ judges whether inputs are the **_real_** samples from the domain $$X_i$$ or **_fake (translated)_** samples, i.e., $$\widetilde{X}_1^2$$ or $$\widetilde{X}_2^1$$. At the same time, each discriminator $$D_i$$ also gives class predictions for input samples, regardless of whether they are real or fake, similar to the one used in AC-GAN [(Odena et al., 2017)](#odena2017).

#### <a name="shared"></a> *Shared Latent Space

| <img src="{{ site.baseurl }}/assets/images/shared_latent.png" alt="Shared Latent" width="690"/><br><em>Source: Liu et al. (2017)</em> | Since the latent space $$Z$$ is shared by both domains, it is possible to generate a target domain sample $$\widetilde{x}_2$$ from a latent vector $$z$$ that was encoded from a source domain sample $$x_1$$, e.g., $$\widetilde{x}_2=G_2(E_1(x_1))$$, or in the opposite direction. |

All these modules are implemented as neural networks.

## Training
In this section, we describe how the image-to-image translation model is trained.

### Preliminaries
Should be added

### Loss and Objective Functions
There are three different types of loss functions used in training of our image-to-image translation models. Each loss function has its own objective according to the role of the corresponding modules in the entire model.

#### VAE Loss
First of all, the VAE losses are deployed to train the encoders $$E_i$$ and the decoders/generators $$G_i$$ to be able to reconstruct the samples from its own domain dataset $$X_i$$ using the stochastic samples $$z$$ from the shared latent space $$Z$$.
The VAE loss function for each domain can be written as follows: 

$$\mathcal{L}_{\text{VAE}_1}(E_1, G_1) = \lambda_1 \text{KL}(q_1(z_1|x_1) \Vert p_\eta (z)) - \lambda_2 \mathbb{E}_{z_1 \sim q_1(z_1|x_1)}[\log p_{G_1} (x_1 | z_1)]$$

$$\mathcal{L}_{\text{VAE}_2}(E_2, G_2) = \lambda_1 \text{KL}(q_2(z_2|x_2) \Vert p_\eta (z)) - \lambda_2 \mathbb{E}_{z_2 \sim q_2(z_2|x_2)}[\log p_{G_2} (x_2 | z_2)]$$

where the hyper-parameters $$\lambda_1$$ and $$\lambda_2$$ control the balance between the reconstruction error and the KL divergence of the latent distribution from its prior.

#### GAN Objective
Next, GAN objectives are used to enforce the translated images look like images from the target domain through the adversarial training of generators and discriminators. For example, $$D_1$$ tries to discriminates the real samples $$x_1 \sim X_1 $$ and the translated fake samples $$\widetilde{x}_1^2 = G_1(z_2)$$. On the other hand, $$G_1$$ tries to make $$D_1$$ classify $$\widetilde{x}_1^2$$ as real samples. It can be formulated by following equations:

$$
\mathcal{L}_{\text{GAN}_1}(E_2, G_1, D_1) = \mathbb{E}_{x_1 \sim P_{X_1}}[\log D_{1s}(x_1)] + \mathbb{E}_{z_2 \sim q_2(z_2|x_2)}[\log (1 - D_{1s}(G_1(z_2)))]
$$

$$
\mathcal{L}_{\text{GAN}_2}(E_1, G_2, D_2) = \mathbb{E}_{x_2 \sim P_{X_2}}[\log D_{2s}(x_2)] + \mathbb{E}_{z_1 \sim q_1(z_1|x_1)}[\log (1 - D_{2s}(G_2(z_1)))]
$$

where each $$D_{is}$$ refers to the discrimination output of $$D_i$$.

#### Classification Loss
Finally, we also introduce the classification losses for the labeled samples, if any, to encourage matching the classes of the samples between two domains. For example, a translated sample $$\widetilde{x}_1^2 = G_1(E_2(x_2))$$ where $$x_2$$ has its label $$y_2$$ in the domain $$X_2$$ should be classified by $$D_1$$ as the same class in the domain $$X_1$$. It can be formulated by cross-entropy loss or negative log-likelihood: 

$$
\begin{align}
\mathcal{L}_{\text{CLASS}_1}(E_2, G_1, D_1) &= \alpha_1 \cdot {\underset { (x_1, y_1) \sim P_{X_{1}^{\text{labeled}}} }{\operatorname {\mathbb{E}} }}[−\log D_{1c}(y_1|x_1)] + \alpha_2 \cdot {\underset { z_2 \sim q_2(z_2|x_2), (x_2, y_2) \sim P_{X_{2}^{\text{labeled}}} }{\operatorname {\mathbb{E}} }}[−\log D_{1c}(y_2|G_1(x_2))]\\
\mathcal{L}_{\text{CLASS}_2}(E_1, G_2, D_2) &= \alpha_2 \cdot {\underset { (x_2, y_2) \sim P_{X_{2}^{\text{labeled}}} }{\operatorname {\mathbb{E}} }}[−\log D_{2c}(y_2|x_2)] + \alpha_1 \cdot {\underset { z_1 \sim q_1(z_1|x_1), (x_1, y_1) \sim P_{X_{1}^{\text{labeled}}} }{\operatorname {\mathbb{E}} }}[−\log D_{2c}(y_1|G_2(x_1))]
\end{align}
$$

where $$\alpha_i$$

$$
\alpha_i = \eta \cdot \frac{\text{the number of all samples in } X_i}{\text{the number of labeled samples in } X_i}
$$

and $$\eta$$ is a hyper-parameters that controls the weight of classification loss.

### Joint Optimization
Combining the losses and objective functions above together, we jointly optimize the following minimax problem:

$$
\begin{align}
{\underset { E_1, E_2, G_1, G_2 }{\operatorname { min } }} \;\; {\underset { D_1, D_2 }{\operatorname { max } }} \quad & \mathcal{L}_{\text{VAE}_1}(E_1, G_1) + \mathcal{L}_{\text{GAN}_1}(E_2, G_1, D_1) + \mathcal{L}_{\text{CLASS}_1}(E_2, G_1, D_1)\\ + &\mathcal{L}_{\text{VAE}_2}(E_2, G_2) + \mathcal{L}_{\text{GAN}_2}(E_1, G_2, D_2) + \mathcal{L}_{\text{CLASS}_2}(E_1, G_2, D_2)
\end{align}
$$

We use an alternating training procedure to solve this. Specifically, we first update $$D_1$$ and $$D_2$$ by applying a gradient ascent step while the parameters of the other modules are fixed. Then, $$E_1, E_2, G_1,$$ and $$G_2$$ are updated with a gradient descent step while $$D_1$$ and $$D_2$$ are fixed.

# Experiments

For our experiment, we utilized our convolutional VAE that we created. We also used a baseline models for Kannada-MNIST datasets as well as a classification model of MNIST data. First, we ran our convolutional VAE model on the Kannada MNIST dataset that we retrieved from Kaggle. From the convolutional VAE, we obtained accuracy values that were then compared with the accuracy values of our baseline models for Kannada-MNIST data and MNIST data. This allowed us to ultimately determine whether our model that we created could translate Kannada numerical values effectively.

## Baselines

**_Sungtae will review this again_**

We compared the results of our classification model to a baseline Kannada-MNIST model. The baseline model was a convulutional neural network with convolutional layers that had increasing output filter sizes (from 32 to 256), a dropout layers with a rate of 0.5 for each convolutional layer, a flatten layer, and a dense layer of 512x10 units. The baseline model showed us how accurately it could evaluate both Kannada-MNIST data and Dig-MNIST data. This model was a baseline. Therefore, it didn't have any changes/differences to how it was evaluating these datasets. It simply was taking in either Kannada-MNIST data or Dig-MNIST data and determining how accurately the model was classifying the test data. The accuracy of this baseline data can be used to compare with the accuracy we get from our MNIST classification model. This is because our MNIST classification model is classifying MNIST data that we obtained from our own CVAE implementation whereas the baseline model is classifying data we had gotten from another dataset.

We compared the results of our classification model to a baseline Kannada-MNIST model. The baseline model was a convulutional neural network with the following layers:
* convolutional layers that had increasing output filter sizes (from 32 to 256)
* a dropout layers with a rate of 0.5 for each convolutional layer
* flatten layer
* a dense layer of 512x10 units

The classification model for MNIST data takes in the MNIST data obtained from the VAE model and creates a convolutional neural network with the following layers:
* convolutional layers that had increasing output filter sizes (from 32 to 256)
* leaky relu layer
* dropout layer
* flatten layer
* dense layer with dimensions 512x10

The baseline models showed us at what accuracy image to image translation should perform in order to be effective for both MNIST and Kannada-MNIST data. We used the baseline model of Kannada-MNIST data as the baseline model for Dig-MNIST data as well. The accuracy of this baseline data can be used to compare with the accuracy we get from our CVAE implementation.

## Results
Our results are depicted visually below. We have shown the loss curve of the CVAE implementation to show that our model is of good fit. We also visually show the translation between KMNIST and MNIST data from MNIST to KMNIST. Our classification performance, which compares the accuracy of each of our models, is also shown below. Finally, we depicted the shared latent space of each of the numerical digits in Kannada and the regular English digits.

### Loss Curve
The CVAE obtained a loss function that is displayed below. This graph shows both the training and validation loss that was obtained. The first graph shows the loss for the MNIST dataset and the second graph shows the loss for the Kannada MNIST dataset:

| MNIST | Kannada |
|:-:|:-:|
| <img src="{{ site.baseurl }}/assets/images/mnist_loss.png" width="370" height="320" /> | <img src="{{ site.baseurl }}/assets/images/kannada_loss.png" width="370" height="320" /> |

For both of these loss functions, we see that our model isn't overfitting nor underfitting. This means that our model can learn from a variety of datasets and can use what it has learned to evaluate generalized data.

### Translation
Below is a visual representation of our translation between Kannada numerals and Arabic numerals. As you can see, with our MNIST data as input, we translate the data through reconstruction and obtain an output of Kannada-MNIST data. With Kannada-MNIST data as an input, we can translate to obtain MNIST data. We show a 3 step process for the translation to and from Kannada numbers. Our first step is to obtain the input data. This input data is represented as either Arabic numerals (MNIST) or Kannada numerals (KMNIST). The second step is the reconstruction step. In this step, the model learns how to recreate our input data and create new data that matches the input data. Finally, the last step is the translation itself. This is where we utilize our CVAE to translate MNIST data to KMNIST data, and vice versa.

| Dataset | Input $$X_i$$ | Reconstruction $$X_i \rightarrow \widetilde{X}_i$$ | Translation $$X_i \rightarrow \widetilde{X}_j$$|
|:-:|:-:|:-:|:-:|
| MNIST   | <img src="{{ site.baseurl }}/assets/images/input_1.gif" width="320" height="320" /> | <img src="{{ site.baseurl }}/assets/images/recon_1_1.gif" width="320" height="320" /> | <img src="{{ site.baseurl }}/assets/images/trans_1_2.gif" width="320" height="320" /> |
| Kannada | <img src="{{ site.baseurl }}/assets/images/input_2.gif" width="320" height="320" /> | <img src="{{ site.baseurl }}/assets/images/recon_2_2.gif" width="320" height="320" /> | <img src="{{ site.baseurl }}/assets/images/trans_2_1.gif" width="320" height="320" /> |

### Classification Performance

We obtained 5 different accuracy values for each of our datasets. For unsupervised, 1% semi-supervised, 5% semi-supervised, 10% semi-supervised, and fully supervised learning, our translation of MNIST data has a high accuracy of about 98%.

The accuracy of our KMNIST data translation increases from unsupervised learning to fully supervised learning. Using unsupervised learning for KMNIST translation, our accuracy was about 3%, showing that our model couldn't translate data effectively. With a 1% semi-supervised learning, the accuracy increased tremendously for KMNIST data translation -- the accuracy was about 88%. This shows that with a small amount of labeled data, our model can translate at a greater performance. This is further shown with the increase in accuracy for 5% semi-supervised learning (accuracy is about 91%) and for 10% semi-supervised learning (accuracy is about 94%). The highest accuracy achieved with KMNIST data translation was about 96% accuracy. This was achieved with fully-supervised learning. Therefore, with labeled data, our model can translate Kannada MNIST data better.

The accuracy of the Dig MNIST dataset also shows an increasing trend as the learning becomes more supervised. For unsupervised learning, the accuracy is about 3%. For 1% semi supervised, the accuracy found was about 59%. For 5% semi supervised, the accuracy is about 67%. For 10% semi supervised, teh accuracy is about 70%. Finally, for fully supervised learning the accuracy is about 75%. Though we have shown that with more labeled data our model can have a higher accuracy with translating Dig MNIST, the highest accuracy obtained was only about 75%, which means that there are still some errors in translation for Dig MNIST.

The Fowlkes-Mallows score is another evaluation metric that we used to show how well our model performed. It shows the similarity among the clusters that are obtained after multiple clustering algorithms have been run. For the MNIST data set, we see that that Fowlkes-Mallows score stays relatively constant at around 0.96, showing that the similarity of clustering is roughly the same for each of the types of learning that we ran our model with. However, for both the KMNIST and Dig-MNIST data, the Fowlkes-Mallows score increases as the learning changes from unsupervised to fully supervised. This means that the similarity in clusterings increases as there is more labeled data to learn from in the model. With more similar clusterings, it is easier to determine what translates to what. Therefore, it makes sense that as the amount of similarity between clustering increases, so does the trend in accuracy.

| ![Accuracy]({{ site.baseurl }}/assets/images/accuracy.png) | ![FMS]({{ site.baseurl }}/assets/images/fms.png) |

### Visualization of the Shared Latent Space

What the visualization below shows is the shared latent space. Latent space helps find a relationship between 2 different domains so that transformations can occur between those 2 domains. When there is a shared latent space, we know that those 2 domains can basically be translated from one to another. What the visualization below shows is that the shared latent space found is between the corresponding digits of Kannada and Arabic numbers. For the most part, each domain in Arabic has a corresponding domain in Kannada that is correct -- the 1s, 2s, 3s, 4s, 5s, 7s, 8s, and 9s match each other. There is no shared latent space between each languages 0s and 6s. The lack of shared latent space between these 2 numerals is a possible reason as to why the accuracy of translation between MNIST and KMNIST isn't 100%.

<!--|       |
|:-:|
| ![tSNE Plot]({{ site.baseurl }}/assets/images/tsne.png) |
-->
<p align="center">
    <img src="assets/images/tsne.png" alt="tSNE" />
    <br>
    <em>tSNE Plot</em>
</p>



# Conclusion
concluding remarks

# Contributions
- Anh: Led the study and discussion about VAE
- Naman: Led the study and discussion about GAN
- Nitya: Designed and implemented MNIST classification models
- Joshua: Designed and implemented Kannada-MNIST classification models 
- Sungtae: Designed and implemented image-to-image translation models  
- Everyone has equally contributed to web page creation

# References
<a name="liu2017"></a>[(Liu et al., 2017) Liu, Ming-Yu, Thomas Breuel, and Jan Kautz. "Unsupervised image-to-image translation networks." Advances in neural information processing systems. 2017.](http://papers.nips.cc/paper/6672-unsupervised-image-to-image-translation-network "UNIT")

<a name="odena2017"></a>[(Odena et al., 2017) Odena, Augustus, Christopher Olah, and Jonathon Shlens. "Conditional image synthesis with auxiliary classifier gans." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.](https://dl.acm.org/doi/10.5555/3305890.3305954 "AC-GAN")

# Cheat Sheet

Text can be **bold**, _italic_, ~~strikethrough~~ or `keyword`.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### My image

![Blacktocat]({{ site.baseurl }}/assets/images/blacktocat.png)

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE4ODA2NzMzXX0=
-->
