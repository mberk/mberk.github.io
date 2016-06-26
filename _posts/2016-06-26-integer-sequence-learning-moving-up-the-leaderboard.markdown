---
title:  "Integer Sequence Learning - Moving Up the Leaderboard"
date:   2016-06-26 09:00:00
description: Parsimony, Parallelism and Actual Machine Learning
---

This post follows on from the [previous one]({% post_url 2016-06-08-integer-sequence-learning-first-steps %}).
If you've not yet read that then I suggest you start there but for a quick recap,
the [Integer Sequence Learning Kaggle competition](https://www.kaggle.com/c/integer-sequence-learning)
is focused on predicting the next element in a sequence of integers such as

$$1, 1, 2, 3, 5, 8, 13, 21, 34, 55$$

In my [last post]({% post_url 2016-06-08-integer-sequence-learning-first-steps %})
I described a simple approach to this problem using a linear model to fit
the sequence and an in-sample measure of goodness of fit to decide whether to
stick with that prediction or fall back to a simpler method such as using the
mode of the sequence. Depending on the order of the linear model (how many
previous values of the sequence to base the prediction on) and the threshold
used to fall back on using the alternative method, you can score about 18%
accuracy which at the time of writing will get you into the top ten of the
leaderboard.

In this post I'm going to discuss three improvements I've made to the approach
which, when combined, boosts the accuracy to 19.57%, currently good enough for
second place. The three improvements are:

- [Adhering to the principle of model parsimony to avoid overfitting](#model-parsimony)
- [Using parallel programming to speed up development](#parallel-programming)
- [Some first steps at combining the linear model with a true machine learning approach](#actual-machine-learning)

Before I begin, a quick note that I'd previously mistakenly communicated that the
test set in this competition is identical to the training set except with the last
element removed. This is, in fact, not the case and the two data sets contain
disjoint sets of sequences.

With that out of the way, let's first consider the issue of model parsimony.

# Model Parsimony

Model parsimony is the idea that simpler models are best. More complicated
models with more parameters are easier to overfit and may end up identifying
spurious relationships in the data. Fitting linear models to sequences is
essentially the same as identifying [recurrence relations](https://en.wikipedia.org/wiki/Recurrence_relation)
and it seems likely that either a sequence has a relatively simple recurrence
relation or it doesn't have one at all.

If we think about the underlying data
generation process in this competition, it's important to realise that there
is some kind of meaning behind every sequence whether it's the ["number of
trees with n unlabelled nodes"](https://oeis.org/A000055) or the ["low-temperature
partition function expansion for Kagome net (Potts model, q=4)"](https://oeis.org/A057405).
I'm not suggesting that you should be able to understand the mathematics
behind every sequence in the data set and I definitely don't but the point is
that someone has been able to reduce each sequence to some succinctly expressed
concept.

Of course, models cannot be _too_ simple otherwise they are unable to capture
the complexities of the data and make poor predictors. The bottom line is that
there is a balance to be struck between goodness of fit and model complexity.
I imagine most readers are already famiiliar with this idea as the problem and
importance of, for example, cross-validation are well understood in
machine learning circles.

Here's how we can apply the idea to build on the code I discussed in my last
post. Here's the original model fitting function:

{% highlight r %}
# Fits a linear model based on the previous <numberOfPoints> elements in the sequence
# If <forSubmission> is true then all of the data will be used and a vector of predictions suitable for submission will be returned
# Otherwise the last element of the sequence will be held back as an out of sample test and a data frame summarising the fit will be returned
# <modeFallbackThreshold> is used when generating submissions to determine when to fall back on using the mode as the prediction
fitModel <- function(sequence,numberOfPoints,forSubmission=FALSE,modeFallbackThreshold)
{
  # Convert to a vector of numbers
  sequence <- as.numeric(strsplit(sequence,split=",")[[1]])
  if(!forSubmission)
  {
    oos <- tail(sequence,1)
    sequence <- head(sequence,-1)
  }
  
  # Need at least <numberOfPoints>+1 observations to fit the model, otherwise just return the last value
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
  else
  {
    df <- data.frame(y=tail(sequence,-numberOfPoints))
    for(i in 1:numberOfPoints)
    {
      df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints+i-1)]
    }

    fit <- lm(y~.,df))
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
{% endhighlight %}

I found two obvious improvements to make here related to the concept of model
parsimony. The first improvement is focused on using too simple a model - as it
stands if the sequence has less than `numberOfPoints+1` elements then the
linear model is not fit and the prediction is simply the last element of the
sequence. A better idea would be to try to still fit a linear model but using
whatever data is available. First, add an argument to the function:

{% highlight r %}
fitModel <- function(sequence,
                     numberOfPoints,
                     forSubmission=FALSE,
                     modeFallbackThreshold,
                     useAllAvailable=FALSE)
{% endhighlight %}

If `useAllAvailable` is `FALSE` then the function should behave in the same way,
otherwise we'll still fit a linear model but one with less inputs. So we'll add
in the following check

{% highlight r %}
# Need at least <numberOfPoints>+1 observations to fit the model, otherwise just return the last value
if(useAllAvailable && length(sequence)<=numberOfPoints && length(sequence)>1)
{
  numberOfPoints <- length(sequence) - 1
}
{% endhighlight %}

This `if` statement checks whether the sequence is shorter than the specified
`numberOfPoints`, that the sequence is still at least 2 or more elements long
and that the `useAllAvailable` option has been set. If all three conditions are
`TRUE` then the `numberOfPoints` variable is modified to be all available data
points. No further modifications to the function are required as it's
essentially the same as if we called the function with a smaller `numberOfPoints`
in the first place.

The second improvement is that rather than go ahead and try to fit a linear
model with `numberOfPoints` straight away, we should first check if the
sequence is well described with a smaller `numberOfPoints`. In other words,
whether a simpler model than the one proposed fits well. If it does then we'd
prefer to use that one for the reasons explained above. We need to modify the
part of the function where we're constructing the data frame and fitting the
linear model. Here's the new code:

{% highlight r %}
df <- data.frame(y=tail(sequence,-numberOfPoints))
for(i in 1:numberOfPoints)
{
  df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints+i-1)]
}
rank <- qr(model.matrix(y~.,df))$rank
if(rank < numberOfPoints)
{
  numberOfPoints0 <- rank
  df <- data.frame(y=tail(sequence,-numberOfPoints0))
  for(i in 1:numberOfPoints0)
  {
    df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints0+i-1)]
  }
  fit <- lm(y~.-1,df[1:numberOfPoints0,])
  mae <- max(abs(predict(fit,df)-df$y))
  if(mae > 1e-4)
  {
    df <- data.frame(y=tail(sequence,-numberOfPoints))
    for(i in 1:numberOfPoints)
    {
      df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints+i-1)]
    }
  }
  else
  {
    numberOfPoints <- numberOfPoints0
    df <- df[1:numberOfPoints,]
  }
}

fit <- lm(y~.-1,df)
{% endhighlight %}

The first few lines are the same. The first new line is 

{% highlight r %}
rank <- qr(model.matrix(y~.,df))$rank
{% endhighlight %}

Without getting into the details of the linear algebra, this is calculating
the [rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra)) of the matrix of
predictors. This rank is an indication of the order of the recurrence relation
that defines the sequence (if there is one). We can therefore check if the rank
is smaller than `numberOfPoints` to see if it's likely that a simpler model based
on less data points is appropriate.

If the check is passed we try fitting this simpler model by temporarily defining a
new `numberOfPoints0`, constructing a new data matrix and fitting the linear
model. We check the Maximum Absolute Error (`mae`) of this proposed model and
either go back to the original `numberOfPoints` and data set if it's too large
or keep the temporary one and proceed to execute the rest of the function as
before.

You'll notice that the threshold for deciding whether this simpler model is
preferable is really conservative (`1e-4`). I chose this value because as I
described at the start of this section, it's likely that either a recurrence
relation exists for a sequence and so the linear model approach works or it
doesn't. If the recurrence relation exists then we should expect the linear model
to have a very small error (explained entirely by numerical rounding issues). I
am guessing that there are still times when the linear model gets correct
predictions even if the fit is not perfect because there are probably some
sequences which can be _approximated_ by linear models but we're talking about
different goals here:

* If a recurrence relation exists we should find it by using the simplest model possible
* If a recurrence relation does not exist then we want to use the highest order linear model possible to try to approximate the sequence as best we can

For clarity, the re-written function in its entirety:

{% highlight r %}
# Fits a linear model based on the previous <numberOfPoints> elements in the sequence
# If <forSubmission> is true then all of the data will be used and a vector of predictions suitable for submission will be returned
# Otherwise the last element of the sequence will be held back as an out of sample test and a data frame summarising the fit will be returned
# <modeFallbackThreshold> is used when generating submissions to determine when to fall back on using the mode as the prediction
fitModel <- function(sequence,
                     numberOfPoints,
                     forSubmission=FALSE,
                     modeFallbackThreshold,
                     useAllAvailable=FALSE)
{
  # Convert to a vector of numbers
  sequence <- as.numeric(strsplit(sequence,split=",")[[1]])
  if(!forSubmission)
  {
    oos <- tail(sequence,1)
    sequence <- head(sequence,-1)
  }
  
  # Need at least <numberOfPoints>+1 observations to fit the model, otherwise just return the last value
  if(useAllAvailable && length(sequence)<=numberOfPoints && length(sequence)>1)
  {
    numberOfPoints <- length(sequence) - 1
  }

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
  else
  {
    df <- data.frame(y=tail(sequence,-numberOfPoints))
    for(i in 1:numberOfPoints)
    {
      df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints+i-1)]
    }
    rank <- qr(model.matrix(y~.,df))$rank
    if(rank < numberOfPoints)
    {
      numberOfPoints0 <- rank
      df <- data.frame(y=tail(sequence,-numberOfPoints0))
      for(i in 1:numberOfPoints0)
      {
        df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints0+i-1)]
      }
      fit <- lm(y~.-1,df[1:numberOfPoints0,])
      mae <- max(abs(predict(fit,df)-df$y))
      if(mae > 1e-4)
      {
        df <- data.frame(y=tail(sequence,-numberOfPoints))
        for(i in 1:numberOfPoints)
        {
          df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints+i-1)]
        }
      }
      else
      {
        numberOfPoints <- numberOfPoints0
        df <- df[1:numberOfPoints,]
      }
    }

    fit <- lm(y~.-1,df)
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
{% endhighlight %}

These two improvements look to be worth around 0.4% accuracy.

# Parallel Programming

The next set of improvements to discuss do not directly yield any increases in
leaderboard score but should reap dividends over the course of the competition.
Parallel programming is a great way to speed up data analysis with little effort
and less time spent waiting for models to fit or features to be constructed is time
that can be spent exploring new ideas. I'm only going to briefly discuss it in this
post but it'll be a recurring theme in future posts as efficiency in the data
analysis pipeline is a topic that's close to my heart.

There are a lot of different options for parallelising computation but a simple one
that's applicable here is to fit models to multiple sequences in parallel. In the
[last post]({% post_url 2016-06-08-integer-sequence-learning-first-steps %}) I gave
the following code for generating a submission:

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

Here are the required changes to speed up this function using parallel programming.
Firstly, we need to import the following packages:

{% highlight r %}
library(plyr)
library(foreach)
library(parallel)
library(doParallel)
{% endhighlight %}

I've previously sung the praises of the [plyr](https://cran.r-project.org/web/packages/plyr/index.html)
package and, indeed, the rest of [Hadley Wickham](http://hadley.nz/)'s work but it's
worth reiterating here. The package offers replacements to the set of `*apply`
functions in R and one of its benefits is an easy way to switch from sequential to
parallel processing. It needs the `foreach` package to enable this functionality
and that, in turn, needs a backend, one of which is provided by the combination of
the `doParallel` and `parallel` packages.

With these packages imported, a rewritten version of the `generateSubmission`
function looks like:

{% highlight r %}
generateSubmission <- function(filename,
                               numberOfPoints,
                               modeFallbackThreshold,
                               useAllAvailable=FALSE,
                               useParallel=FALSE)
{
  if(useParallel)
  {
    registerDoParallel(8)
  }
  
  submission <- data.frame(Id=data$Id,
                           Last=laply(1:nrow(data),
                                      function(i)
                                      {
                                        model <- fitModel(
                                          data$Sequence[[i]],
                                          numberOfPoints=numberOfPoints,
                                          modeFallbackThreshold=modeFallbackThreshold,
                                          useAllAvailable=useAllAvailable,
                                          forSubmission=TRUE)
                                        return(model)
                                      },.parallel=useParallel))
  
  if(useParallel)
  {
    stopImplicitCluster()
  }
  
  options(scipen=999)
  write.csv(submission,filename,row.names=FALSE)
}
{% endhighlight %}

There's now a `useParallel` option in the argument list. If this is enabled, the first
step is to call `registerDoParallel(8)`. This function, provided by the `doParallel`
package, sets up a cluster of R processes on your local machine, ready to carry out
parallel computation on demand. The number passed in as an argument is the number of
these worker processes to create and you should be careful here to choose a value that
is appropriate for the number of cores on your machine and your available memory. Most
modern PCs have between 4 and 8 cores but bear in mind that each worker may need its
own copy of the entire data set and so memory usage could end up being more of a
limiting factor than cores. I'll go into these considerations in more detail in a
future post.

The next key change is from using `sapply` to `laply` when fitting the model to each
sequence. This is one of the `*apply` replacements in the `plyr` package. For this
particular function the format of the arguments in the same as `sapply` but there is
an additional argument `.parallel`. When this is set to `TRUE` then `plyr` will
automatically handle parallelisation of the processing you've given it, provided
you've set up the `foreach` backend, as we did with the call to `registerDoParallel`.

The only other change is to shutdown the cluster that was set up by calling
`stopImplicitCluster`.

In terms of bottom line benefits, on my laptop with 8 cores I can generate a
submission based on `numberOfPoints=12` in 11.75 minutes without parallelisation
and in 3.5 minutes with it. You may have picked up on the fact that despite having 8
worker processes at its disposal, the parallel version is not even close to 8 times
faster. This'll be down to overheads and variation in the computational complexity
of processing different sequences (shorter ones will be less work than longer ones,
for instance). Nevertheless, it _is_ over three times faster and those 8 or so
minutes saved are time you can be spending on the next version of the model. If you
consider that you can be making up to five submissions per day, the benefits
rapidly start to compound.

# Actual Machine Learning

The approach I've taken to this competition so far cannot really be described as
machine learning - each sequence is fit in isolation and no infomation is shared
across sequences. 

A simple idea I had in this regard is to look at all pairs of consecutive elements
across all sequences both in the training and test sets. In this way, a kind of
frequency table can be constructed which, for any given integer gives the most
likely integer to come next.

This method is a good candidate to replace the fallback approach where I predict
the mode of the sequence as the next element when the linear model does not
describe the sequence well. We may still need to resort to the mode prediction
if the final element does not exist in the frequency table but it seems likely
that where it can be used the frequency table will be more effective as it
leverages information from all sequences in the data set.

Practically speaking, here is my function for building the frequency table:

{% highlight r %}
buildFrequencyTable <- function(sequences,
                                useParallel=FALSE)
{
  if(useParallel)
  {
    registerDoParallel(8)
  }
  
  pairs <- ldply(sequences,
                 function(sequence)
                 {                 
                   sequence <- as.numeric(unlist(strsplit(sequence,",")))
                   if(length(sequence) < 2)
                   {
                     return(NULL)
                   }
                   data.frame(last=head(sequence,-1),
                              following=tail(sequence,-1))
                 },
                 .parallel=useParallel)

  frequencyTable <- ddply(pairs,
                          "last",
                          function(x)
                          {
                            data.frame(last=x$last[1],
                                       following=Mode(x$following))
                          },
                          .parallel=useParallel)

  if(useParallel)
  {
    stopImplicitCluster()
  }

  return(frequencyTable)
}
{% endhighlight %}

Straight away you'll notice that I'm using the same parallel processing
framework that I discussed in the previous section. The method for building the
frequency table consists of two stages - first each sequence is broken down into
pairs of consecutive integers. Then, for each unique integer across all sequences,
the most frequent following integer is found using the existing `Mode` function.

Note how I build the pairs using the `ldply` function. The functions provided by
the `plyr` package follow a naming convention where the first character
corresponds to the input type and the second character corresponds to the output
type. In this case we have `l` as the first character, standing for `list` and
`d` as the second character, standing for `data.frame`. So `ldply` takes a list
as input, applys a function to each element, and produces a `data.frame` as
output. So the function that's passed as the second argument should return a
`data.frame` and `ldply` will handle assembling the individual `data.frame`s into
a single one. A final point of interest here is that if the sequence consists of
only a single element, so it's impossible to know what comes next, I
return `NULL`. `ldply` will discard any `NULL`s that the
function output when it assembles the final `data.frame`.

Next see that I use the `ddply` function to construct the frequency table. As you
might have guessed, this function takes a `data.frame` as input and also outputs
a `data.frame`. However you can see that it takes a third argument between the
input and the function, in this case the string `"last"`. This argument specifies
how `ddply` should divide up the `data.frame` input. In other words, I'm telling
it to split the input `data.frame` into a series of individual `data.frame`s, one
for each unique value of the `"last"` variable and then apply the function to
each of those `data.frames`. The final output will be one big `data.frame` with
two columns where each row corresponds to a unique integer across all sequences
in the data set. The "last" column will be that unique integer and the
"following" column will be the most likely integer that comes after it.

Finally, here's a modified version of the `fitModel` function that makes use of
the frequency table:

{% highlight r %}
# Fits a linear model based on the previous <numberOfPoints> elements in the sequence
# If <forSubmission> is true then all of the data will be used and a vector of predictions suitable for submission will be returned
# Otherwise the last element of the sequence will be held back as an out of sample test and a data frame summarising the fit will be returned
# <modeFallbackThreshold> is used when generating submissions to determine when to fall back on using the mode as the prediction
fitModel <- function(sequence,
                     numberOfPoints,
                     useAllAvailable=FALSE,
                     forSubmission=FALSE,
                     modeFallbackThreshold,
                     frequencyTable=NULL)
{
  # Convert to a vector of numbers
  sequence <- as.numeric(strsplit(sequence,split=",")[[1]])
  if(!forSubmission)
  {
    oos <- tail(sequence,1)
    sequence <- head(sequence,-1)
  }
  
  # Need at least <numberOfPoints>+1 observations to fit the model, otherwise just return the last value
  if(useAllAvailable && length(sequence)<=numberOfPoints && length(sequence)>1)
  {
    numberOfPoints <- length(sequence) - 1
  }
  
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
  else
  {
    df <- data.frame(y=tail(sequence,-numberOfPoints))
    for(i in 1:numberOfPoints)
    {
      df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints+i-1)]
    }
    rank <- qr(model.matrix(y~.,df))$rank
    if(rank < numberOfPoints)
    {
      numberOfPoints0 <- rank
      df <- data.frame(y=tail(sequence,-numberOfPoints0))
      for(i in 1:numberOfPoints0)
      {
        df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints0+i-1)]
      }
      fit <- lm(y~.-1,df[1:numberOfPoints0,])
      mae <- max(abs(predict(fit,df)-df$y))
      if(mae > 1e-4)
      {
        df <- data.frame(y=tail(sequence,-numberOfPoints))
        for(i in 1:numberOfPoints)
        {
          df[[paste0("x",i)]] <- sequence[i:(length(sequence)-numberOfPoints+i-1)]
        }
      }
      else
      {
        numberOfPoints <- numberOfPoints0
        df <- df[1:numberOfPoints,]
      }
    }

    fit <- lm(y~.-1,df)
    mae <- max(abs(fit$residuals))

    # Make prediction
    if(forSubmission && mae > modeFallbackThreshold)
    {
      if(!is.null(frequencyTable) && tail(sequence,1) %in% frequencyTable$last)
      {
        prediction <- frequencyTable$following[frequencyTable$last==tail(sequence,1)]
      }
      else
      {
        prediction <- Mode(sequence)
      }
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
    results <- data.frame(prediction=prediction,
                          mae=mae,
                          oos=oos,
                          mode=Mode(sequence))
    
    if(!is.null(frequencyTable))
    {
      if(tail(sequence,1) %in% frequencyTable$last)
      {
        results$frequencyPrediction <- 
          frequencyTable$following[frequencyTable$last==tail(sequence,1)]
      }
      else
      {
        results$frequencyPrediction <- NA
      }
    }
    return(results)
  }
}
{% endhighlight %}

I've added a `frequencyTable` argument for passing the table through to the
function. When the function is called to generate a submission and the MAE
threshold for rejecting the linear model is exceeded, I check whether a
frequency table was given and
if so, check whether the last element of the sequence exists in the table. If it
does then the table is used to make the prediction, otherwise the mode is used
as the fallback as usual. When the function is not called in the context of
generating a submission then the frequency table prediction is always included
in the output, assuming a frequency table was given, that way it is available for
further evaluation along with the rest of the model diagnostics.

I estimate that the frequency table approach is worth about 1.1% accuracy compared
to only using the mode as the fallback.

We've covered a lot of ground in today's post and I hope you've found the material
useful. As usual please leave your comments below or contact me via any of the
methods on the [About page]({{ site.baseurl }}about).
