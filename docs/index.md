---
layout: default
---
# StackEdit Working?
- Sungtae - working

# Intro
Brief introduction about our motivation

## MNIST and Kannada-MNIST
Brief description of MNIST and Kannada-MNIST

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

For our experiment, we utilized our convolutional VAE that we created. We also used a baseline models for Kannada-MNIST datasets as well as a classification model of MNIST data. First, we ran our convolutional VAE model on the Kannada MNIST dataset that we retrieved from Kaggle. The convolutional VAE outputs MNIST data, which we used as input in our classification model. The classification model then gave us accuracy values that determined whether our translation model gave us good data. Finally, we compared these accuracy values to the ones we obtained with our baseline Kannada-MNIST model to ultimately determine whether our model that we created could translate Kannada numerical values to Arabic numerical values.

Need to add:
* description of how classification model of MNIST is created

## Baselines

We compared the results of our classification model to a baseline Kannada-MNIST model. The baseline model was a convulutional neural network with convolutional layers that had increasing output filter sizes (from 32 to 256), a dropout layers with a rate of 0.5 for each convolutional layer, a flatten layer, and a dense layer of 512x10 units. The baseline model showed us how accurately it could evaluate both Kannada-MNIST data and Dig-MNIST data. This model was a baseline. Therefore, it didn't have any changes/differences to how it was evaluating these datasets. It simply was taking in either Kannada-MNIST data or Dig-MNIST data and determining how accurately the model was classifying the test data. The accuracy of this baseline data can be used to compare with the accuracy we get from our MNIST classification model. This is because our MNIST classification model is classifying MNIST data that we obtained from our own CVAE implementation whereas the baseline model is classifying data we had gotten from another dataset. 

## Translation

Outline of results:
* loss function of CVAE (explain what it is saying)
* state accuracy of classification model
* state accuracy of baseline model
* compare both accuracies and state what each means
* Display pictures of translation (found in results section)

| Dataset | Input $$X_i$$ | Reconstruction $$X_i \rightarrow \widetilde{X}_i$$ | Translation $$X_i \rightarrow \widetilde{X}_j$$|
|:-:|:-:|:-:|:-:|
| MNIST   | <img src="{{ site.baseurl }}/assets/images/input_1.gif" width="320" height="320" /> | <img src="{{ site.baseurl }}/assets/images/recon_1_1.gif" width="320" height="320" /> | <img src="{{ site.baseurl }}/assets/images/trans_1_2.gif" width="320" height="320" /> |
| Kannada | <img src="{{ site.baseurl }}/assets/images/input_2.gif" width="320" height="320" /> | <img src="{{ site.baseurl }}/assets/images/recon_2_2.gif" width="320" height="320" /> | <img src="{{ site.baseurl }}/assets/images/trans_2_1.gif" width="320" height="320" /> |

## Classification Performance

|        |           |
|:-:|:-:|:-:|
| ![Accuracy]({{ site.baseurl }}/assets/images/accuracy.png)           | ![FMS]({{ site.baseurl }}/assets/images/fms.png) |

## Visualization of the Shared Latent Space

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
