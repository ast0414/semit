
---
layout: default
---
# StackEdit Working?
- Sungtae - working

# Intro
We set out to try a new approach on generalization in machine learning algorithms. It is often hard to train a model on predicting labels for a given set of data that will perform well on new sets of data. Training a new model on new datasets may also be infeasible due difficulties like: a lack of labels or difficulty in collecting new data. With our new approach to generalization, we hope to improve a neural network model's ability to classify to new datasets. To accomplish this, we are using a Variational Autoencoder (VAE) as well as a Generative Adversarial Network (GAN) to achieve accurate image-to-image translation. We use this translation as a form of  domain adaptation. As a proof-of-concept, we explore this model's ability to adapt the lesser known KannadaMNIST dataset to images similar to the more well known MNIST dataset. The resulting images are then classified with a pre-trained MNIST model.

## MNIST and Kannada-MNIST

The MNIST dataset is a commonly used dataset in machine learning. This dataset consists of images of size 28 pixels by 28 pixels. Each image contains a hand-drawn arabic numeral between 0 to 9 inclusive. Typically, this dataset is used to train a supervised machine learning model to predict an arabic numeral label for a given 28 by 28 pixel image. Here is an example of labeled MNIST data:

![MNIST]({{ site.baseurl }}/assets/images/MNIST_labeled.png)

The Kannada-MNIST, or K-MNIST, dataset is similar to the MNIST dataset except the images are of hand-drawn Kannada numerals instead of arabic numerals. For reference, Kannada is a language predominantly spoken in Karnataka (a state in the southwest of India). This dataset is fairly new, and some are still researching on how to train the most accurate model to predict the labels for this dataset. The following is an example of labeled K-MNIST data:

![K-MNIST]({{ site.baseurl }}/assets/images/KMNIST_labeled.png)

# Methods

## What is VAE?
about variational autoencoder


## What is GAN?
GAN stands for Generative Adversarial Network, which are deep-learning based generative models. GANs are a model architecture for training generative models which are widely used to translate inputs from one domain to another. GANs were first introduced in 2014 by Ian Goodfellow et al in a paper titled "[Generative Adversarial Networks]([https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661))" . While initially proposed as a model for unsupervised learning, GANs have also proved to be useful for semi-supervised learning, fully supervised learning and reinforcement learning.

The GAN model involves two sub-models:

1. **Generator Model** - This is a model that is used to generate new examples from the problem domain. 
	
	The input to the model is a vector from a multidimensional space. After training with the dataset, this multidimensional space is mapped to corresponding points in the problem domain. This forms a compressed representation of the multidimensional data space. 

	After training, the generator model is used to generate new samples.
 
2. **Discriminator Model** - This model is used to classify example inputs based on whether they come from the problem domain or from the generated examples. 

	The model inputs an example from the domain (real or generated) and classifies it with a binary label *real* or *fake*. The *real* examples come from the training dataset, while the *fake* examples come from the generator model.

	Typically, after training is complete, the discriminator model is discarded since we are more interested in the generator model to generate more samples.

![Architecture](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/04/Example-of-the-Generative-Adversarial-Network-Model-Architecture.png "Sample Generative Adversarial Network Architecture")


A key use of generative adversarial networks comes in image-to-image translation, to map images from the input domain to a different output domain. 

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
|:-:|:-:|:-:|:-:|
| MNIST | <img src="{{ site.baseurl }}/assets/images/mnist_loss.png" width="370" height="320" /> |
| Kannada   | <img src="{{ site.baseurl }}/assets/images/kannada_loss.png" width="370" height="320" /> |

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

|        |           |
|:-:|:-:|:-:|
| ![Accuracy]({{ site.baseurl }}/assets/images/accuracy.png)           | ![FMS]({{ site.baseurl }}/assets/images/fms.png) |

### Visualization of the Shared Latent Space

What the visualization below shows is the shared latent space. Latent space helps find a relationship between 2 different domains so that transformations can occur between those 2 domains. When there is a shared latent space, we know that those 2 domains can basically be translated from one to another. What the visualization below shows is that the shared latent space found is between the corresponding digits of Kannada and Arabic numbers. For the most part, each domain in Arabic has a corresponding domain in Kannada that is correct -- the 1s, 2s, 3s, 4s, 5s, 7s, 8s, and 9s match each other. There is no shared latent space between each languages 0s and 6s. The lack of shared latent space between these 2 numerals is a possible reason as to why the accuracy of translation between MNIST and KMNIST isn't 100%. 

|       |
|:-:|
| ![tSNE Plot]({{ site.baseurl }}/assets/images/tsne.png) |

# Conclusion
concluding remarks


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
