---
title:  "Integer Sequence Learning - First Steps"
date:   2016-06-08 10:00:00
description: Hello World
---

Let's get something out of the way up front. I'm not really in a position to
teach you how to win Kaggle competitions, as a cursory glance at my
[profile](https://www.kaggle.com/endintears) should make abundantly clear. I
do like to think, however, that I have some small talent for writing and
teaching (please comment below if you violently disagree).

Additionally, I have various qualifications proving that I know something both
about statistics and computer science, and I combined the two long before the
term 'data science' was coined. I like to build complete systems, not just
fitting models, but getting involved in the entire data analysis pipeline from
data processing to visualisation to parameter optimisation to building
dashboards and I love to automate as much of that process as possible. I'm
also passionate about identifying and adhering to best practice such as
version control and reproducible research.

With these ideas in mind, let's take a look at my first steps in a new Kaggle
competition,
[Integer Sequence Learning](https://www.kaggle.com/c/integer-sequence-learning).

Why this particular competition? Firstly, my computer vision skills are
particularly weak and despite being in the midst of
[Udacity's Deep Learning course](https://www.udacity.com/course/deep-learning--ud730)
I don't think I'll have anything useful to contribute to competitions like
[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection).
[Expedia Hotel Recommendations](https://www.kaggle.com/c/expedia-hotel-recommendations)
is almost over and I like to get involved right at the start where it feels like
I can make the biggest contribution with tutorials and discussions before my
machine learning skills are outstripped by more experienced competitors.

I have to say I really like the spirit of
[Shelter Animal Outcomes](https://www.kaggle.com/c/shelter-animal-outcomes/leaderboard)
but a quick read of the forums suggested there was an annoying leak in the
data that would make it hard to tell whether the top ranked competitors
actually had a good model or were just exploiting the leak (intentionally or
otherwise). Furthermore, [Megan Risdal](https://www.kaggle.com/mrisdal) had
already put together a fantastic R notebook
[script](https://www.kaggle.com/mrisdal/shelter-animal-outcomes/quick-dirty-randomforest)
which made me feel like there was little further to add in terms of tutorials.

So, [Integer Sequence Learning](https://www.kaggle.com/c/integer-sequence-learning).
This competition is a strange one. In my mind, the correct solution for a
system that predicts integer sequences is to automate the querying of the
[On-Line Encyclopedia of Integer Sequences (OEIS)](https://oeis.org/). However, as the
OEIS is the exact source of sequences in this competition that would rather
defeat the purpose. With my aforementioned experience with the 
[Shelter Animal Outcomes](https://www.kaggle.com/c/shelter-animal-outcomes/leaderboard)
and the fact that someone is already sitting top of the 
[Integer Sequence Learning](https://www.kaggle.com/c/integer-sequence-learning)
leaderboard with an implausible accuracy of 72.4%, it's clear that these
knowledge competitions are not as well designed as the ones that reward
competitors with cash, jobs, or at the very least, points. It's arguable how
important it really is to improve competitions which are only intended as
playgrounds but you do have to question whether it's even appropriate to have
a public leaderboard for them.

The goal of this particular competition is, given a sequence of integers such
as

$$1, 1, 2, 3, 5, 8, 13, 21, 34, 55$$

to predict the next element in the sequence
(in this case this is the
[Fibonacci Sequence](https://en.wikipedia.org/wiki/Fibonacci_number) and the
answer is 89). You're provided with 113,845 sequences from the
OEIS and the only difference between the training and test
set is that the test set has the final element of each sequence removed, this
being the prediction target. This explains why it would be easy for someone to
accidentally access data they shouldn't while building their model. In the
extreme case you could just use the training set to achieve 100% accuracy on
the leaderboard. For a competition like this one, I think it's safest to simply
never touch the provided training set as you can always remove elements from
the test set sequences to perform your out of sample predictions.

There hasn't been a lot of discussion on the forums about how to approach this
problem although I did reply to a post where I highlighted the
[RATE package by Christian Krattenthaler](http://www.mat.univie.ac.at/~kratt/)
and the related [GUESS by Martin Rubey](http://axiom-wiki.newsynthesis.org/GuessingFormulasForSequences).
The former is written in Mathematica and the latter in Maple and I doubt
either of these are accessible to the average Kaggler. A [recent blog post by
Derek Jones](http://shape-of-code.coding-guidelines.com/2016/06/07/predicting-the-next-value-in-an-integer-sequence/)
mentions that these methods only achieve around 20% accuracy on OEIS data
anyway.

I figured a straight forward initial approach would be to fit a linear model
using the last \\(N\\) elements to predict the next one. \\(N\\) can be optimised
by constructing an out of sample test. The approach can be further improved
by evaluating the model fit before making the prediction and resorting to
another method when the linear model does not describe the sequence well. The
obvious choice for that other method would be to use the mode of the sequence
which is the benchmark for this competition, written in this
[script](https://www.kaggle.com/wcukierski/integer-sequence-learning/mode-benchmark).

I haven't participated in a Kaggle competition for a few months now and the
scripts system is new to me. Previously if I'd written a tutorial or benchmark
beating script I'd just attach it to a forum post, whereas the scripts system
now lets you develop solutions without even leaving your browser. Other users
can then vote and comment on the scripts and fork them as the basis for their
own scripts. There are a few caveats, however, namely that you cannot really
develop a private script - as soon as you run it, it's available for all to
see. But it's important to understand that Kaggle is providing this
functionality to facilitate knowledge transfer amongst participants, not as
a free computing resource.

I'll now talk through my script, written in R, which is available
[here](https://www.kaggle.com/endintears/integer-sequence-learning/linear-models),
although I may have further improved on that by the time you read this.

{% highlight r %}
library(plyr)

Mode <- function(x)
{
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
{% endhighlight %}

[plyr](https://cran.r-project.org/web/packages/plyr/index.html), written by
[Hadley Wickham](http://hadley.nz/), really is a fantastic package. In fact,
Hadley is responsible for some incredible developments in R
and you'd be well advised to check out his other work. In this script I only
use `plyr` to simplify processing a list
and producing a data frame via the `ldply`
function but I use it extensively in most of my other projects. The
`Mode` function defined here was taken from
the
[benchmark script](https://www.kaggle.com/wcukierski/integer-sequence-learning/mode-benchmark).
As pointed out in that script, the function is named
`Mode` rather than
`mode` as the latter is already a function in R.

I'll next discuss the main function of my script, called
`fitModel`, which I'll break into parts
given its length.


{% highlight r %}
fitModel <- function(sequence,numberOfPoints,forSubmission=FALSE,modeFallbackThreshold)
{
  # Convert to a vector of numbers
  sequence <- as.numeric(strsplit(sequence,split=",")[[1]])
  if(!forSubmission)
  {
    oos <- tail(sequence,1)
    sequence <- head(sequence,-1)
  }
{% endhighlight %}

The function is designed to be run in one of two ways, specified via the
`forSubmission` argument. When this is
`TRUE`, the function will return a vector
of predictions which are suitable for submitting to Kaggle. When set to
`FALSE`, the function will return a data
frame with some more detailed diagnostic information based on an out of
sample test constructed by removing the last element of the sequence.

The `numberOfPoints` argument specifies how many points to use to predict the
next element in the sequence. For example, if `numberOfPoints=2` then the
\\((n-2)\\)th and \\((n-1)\\)th element are used to predict the \\(n\\)th
element.

{% highlight r %}
  # Need at least numberOfPoints+1 observations to fit the model
  # Otherwise just return the last value
  if(length(sequence)<=numberOfPoints)
  {
    if(length(sequence)==0)
    {
      prediction <- NA
    }
    else
    {
      prediction <- tail(sequence,1)
    }
    mae <- NA
  }
{% endhighlight %}

To fit the model the sequence needs to be at least
`numberOfPoints+1` long,
`numberOfPoints` for the predictors and
`1` for the response. If the sequence is
empty, which may be the case when the raw sequence is only a single element
and that's been removed to form the out of sample test, then we can't make a
prediction and it gets set to `NA`. If
there is at least one element then a simple prediction is made by taking the
last element of the sequence. In either case, we set
`mae` to 
`NA`. This variable, standing for Maximum
Absolute Error, is used to indicate how well the linear model fits the
sequence. As there is no linear model for a sequence of this length, we set
it to `NA`.

{% highlight r %}
  else
  {
    df <- data.frame(y=tail(sequence,-numberOfPoints))
    for(i in 1:numberOfPoints)
    {
      df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints+i-1)]
    }

    fit <- lm(y~.,df)
{% endhighlight %}

Now we come to the main part of the function. We use the
`lm` function to fit a linear model,
providing a formula describing the model and a data frame that contains the
variables. Note how the formula is specified; normally you'd write out your
predictors on the right hand side of `~`
such as in `lm(sr ~ pop15 + pop75 + dpi + ddpi, data = LifeCycleSavings)`.
However in this case the number of predictors is determined by the 
`numberOfPoints` parameter and so we cannot
write out the formula in advance (we must also construct the data frame
containing the data for fitting the model on the fly in the
`for` loop). By using a
`.` on the right hand side of the formula,
we are specifying that all variables in the data frame given by
`df` should be included in the model.

{% highlight r %}
    mae <- max(abs(fit$residuals))

    # Make prediction
    if(forSubmission && mae > modeFallbackThreshold)
    {
      prediction <- Mode(sequence)
    }
    else
    {
      df <- list()
      for(i in 1:numberOfPoints)
      {
        df[[paste0("x",i)]] <- sequence[length(sequence)-numberOfPoints+i]
      }
      df <- as.data.frame(df)
  
      prediction <- predict(fit,df)
    }

    # Round the prediction to an integer
    prediction <- round(prediction)
  } 
  
  if(forSubmission)
  {
    return(prediction)
  }
  else
  {
    return(data.frame(prediction=prediction,
                      mae=mae,
                      oos=oos,
                      mode=Mode(sequence)))
  }
}
{% endhighlight %}

After fitting the model, we calculate the Maximum Absolute Error (MAE) as in
order to quantify how well it fits the data. Note that this is done in sample
as the goal is to decide whether we want to use the linear model to make our
prediction or try another method.
`modeFallbackThreshold` gives the value of
the MAE above which the linear model is rejected in favour of the mode of the
sequence. If the `forSubmission` argument
is `FALSE`, there's no need to check the
threshold as the function will return both the linear model prediction and the
mode and the relative merits of the two can be assessed with further analysis.

An important step is to `round` the
prediction before returning. `lm` has no
concept of operating only with integers and will produce coefficients and
predictions which are decimals. This may be an interesting avenue to explore
later but it turns out that ignoring the problem and just rounding the model
predictions works well, at least as a first attempt.

{% highlight r %}
evaluateResults <- function(results,modeFallbackThreshold)
{
  (sum((results$prediction==results$oos)[results$mae<modeFallbackThreshold],na.rm=TRUE) +
   sum((results$mode==results$oos)[results$mae>=modeFallbackThreshold],na.rm=TRUE)) /
    sum(!is.na(results$prediction))
}
{% endhighlight %}

Next we come to a couple of helper functions. First is
`evaluateResults` which is intended to be
used with the output from `fitModel` when
`forSubmission` was set to
`FALSE`, calculating the accuracy on the
last element of the sequence that was held back for the out of sample test.
The function takes an additional argument,
`modeFallbackThreshold`, which sets the
value of the MAE at which the mode of the sequence is used as the prediction
instead of the linear model. In this way, given a set of results, the
threshold can be optimised via repeated calls to this function and finding
which values gives the maximum accuracy on the held back samples.


{% highlight r %}
generateSubmission <- function(filename,numberOfPoints,modeFallbackThreshold,verbose=TRUE)
{
  submission <- data.frame(
    Id=data$Id,
    Last=sapply(1:nrow(data),
                function(i)
                {
                  model <- fitModel(data$Sequence[[i]],
                                    numberOfPoints=numberOfPoints,
                                    modeFallbackThreshold=modeFallbackThreshold,
                                    forSubmission=TRUE)
                                    if(verbose && i %% 1000 == 0)
                                    {
                                      print(paste("Done",i,"sequences"))
                                    }
                                    return(model)
                }))
  options(scipen=999)
  write.csv(submission,filename,row.names=FALSE)
}
{% endhighlight %}

The second helper function is
`generateSubmission` which, as the name
suggests, simplifies the process of generating a CSV file that can be
uploaded to Kaggle. It's more or less a wrapper that calls
`fitModel` for every sequence in the data
set. It's worth mentioning here that I'd normally use one of the
`plyr` functions which all take a
`.progress` parameter which displays a
text (`.progress="text"`) or
visual (`.progress="tk"`) progress bar
showing you how far through the processing it is. I found that this didn't
really work with the Kaggle scripts system. Obviously you have to use a text
progress bar as there is no desktop environment but rather than the bar
remaining on one line and gradually filling up, it was instead periodically
displayed on a new line, like this:

```
  |                         
  |                                                                      |   0%
  |                                                                            
  |                                                                      |   1%
  |                                                                            
  |=                                                                     |   1%
  |               
                                                             
  |=                                                                     |   2%
  |                                                                            
  |==                                                                    |   2%
  |                                                                            
  |==                                                                    |   3%
```

I guess this is down to the way the output of the script environment is being
captured. Regardless, it's simple enough to just use the built in
`sapply` and add some output of our own
periodically (as I've set it up here, every 1,000 sequences).

It's also worth highlighting the need to call
`options(scipen=999)` before writing the
CSV file otherwise some of the large values will be output in scientific
notation and you'd lose some of the digits.

That's it for the function definitions. To generate a submission we can run:

{% highlight r %}
data <- read.csv("../input/test.csv",stringsAsFactors=FALSE)
generateSubmission("linearPrevious10WithModeFallback.csv",
                   numberOfPoints=10,
                   modeFallbackThreshold=15)
{% endhighlight %}

Note that when using Kaggle scripts, the files can be found in the
`"../input/"`. This has tripped me up a
few times where I've taken a script I've developed locally, pasted it into
the Kaggle system and forgotten to change the path. Under these
circumstances the script will immediately fail as it cannot find the input
file but because you ran the script a new, broken version will be generated
and this version will be available for all to see until you fix it.

The final thing to show you is how the hyperparameters, namely the number
of points to use in fitting the linear model and the MAE threshold for
resorting to a mode prediction, can be optimised for accuracy:

{% highlight r %}
possibleNumberOfPoints <- 8:10
possibleModeFallbackThresholds <- c(5,10,15,20,25,50,100,250,500,1000)
accuracies <- matrix(NA,
                     nrow=length(possibleNumberOfPoints),
                     ncol=length(possibleModeFallbackThresholds))
for(numberOfPoints in possibleNumberOfPoints)
{
  print(paste("Trying",numberOfPoints,"points"))
  results <- ldply(1:nrow(data),
                   function(i)
                   {
                     model <- fitModel(data$Sequence[[i]],
                                       numberOfPoints=numberOfPoints,
                                       forSubmission=FALSE)
                     if(i %% 1000 == 0)
                     {
                       print(paste("Done",i,"sequences"))
                     }
                     return(model)
                   })

  for(modeFallbackThreshold in possibleModeFallbackThresholds)
  {
    accuracies[match(numberOfPoints,
                     possibleNumberOfPoints),
               match(modeFallbackThreshold,
                     possibleModeFallbackThresholds)] <-
      evaluateResults(results, modeFallbackThreshold)
  }
}
{% endhighlight %}

This is a pretty basic approach and I'm sure some big improvements could be
made here both in terms of efficiency, so that more parameter values can be
tried, and automation so that the optimal combination can be found withoout
having to specify the grid of values to search over in advance.

A quick comment on the use of a `for`
loop here. Many people are quick to argue that you should avoid
`for` loops in R wherever possible and,
believe me, I used to argue it more than most. However it turns out that this
advice is not really true, especially when slavishly replacing the loops with
`*apply` impacts readability. For more
background on why `for` loops are OK check
out this
[Stack Overflow answer](http://stackoverflow.com/questions/7142767/why-are-loops-slow-in-r/7142982#7142982)
and the [article on page 46 of the May 2008 issue of R news](https://www.r-project.org/doc/Rnews/Rnews_2008-1.pdf).

I hope you've found this walk through useful. If you have any questions or
comments, either on this post or the site in general, please leave them below.
