---
title:  "Bosch Production Line Performance - First Steps"
date:   2016-08-28 10:00:00
description: Out of Memory Analysis in R
---

First, a quick update on my progress with the Integer Sequence Learning
competition. It was my original intention to publish blog posts much more
frequently than I have managed to do so so far but unfortunately real life got
in the way these past couple of months. I also found that I was struggling to
find things to write about as I got caught up in an idea that I simply could not
get to work. I realise that negative results can be as interesting if not moreso
than successful ones but in this case I wasn't feeling it. I might write up the
idea in the future but it seems likely I won't be making further contributions
to that particular competition.

In the mean time I've been checking out new competitions posted on Kaggle to see
if anything captured my interest. As the title of this post suggests, it's the
most recent of these, the
[Bosch Production Line Performance](https://www.kaggle.com/c/bosch-production-line-performance)
competition that's managed to do so.

Call me idealistic but it's not the aim of the competition itself that excites
me. I think that big corporations get an incredibly good deal on Kaggle as they
get to leverage the skills of hundreds if not thousands of talented machine
learners for a fraction of the cost if they were doing the analysis in house.
I'm normally far more interested in competitions hosted by academics or
nonprofits or those which are healthcare related.

But it's that same idealism which has drawn me to this competition which claims
to have one of the largest data sets ever hosted on Kaggle. In a
[deservedly highly rated post on the forums by Kaggler Laurae](https://www.kaggle.com/c/bosch-production-line-performance/forums/t/22909/expeditive-exploration-models-on-data),
we can see that the training data consists of 1,183,747 samples with 969 numeric
features, 2,141 categorical features and 1,157 date features.
[Another post by Laurae](https://www.kaggle.com/c/bosch-production-line-performance/forums/t/22908/datasets-size-uncompressed-14-3gb)
tells us that the uncompressed data amounts to 14.3 GB. I think it's fair to say
that most people are unlikely to have enough RAM in their PCs to store this data
in memory, let alone manipulate it. There's a
[post on the forums about using Hadoop or Spark](https://www.kaggle.com/c/bosch-production-line-performance/forums/t/22986/any-recommendation-on-using-hadoop-or-spark-clusters-to-do-the-analysis)
and a couple of other mentions of using
[Amazon Web Services (AWS)](https://aws.amazon.com/) which could be
used either to have on-demand access to a single well resourced machine or a
distributed system on which to run the aforementioned
[Hadoop](http://hadoop.apache.org/) or [Spark](http://spark.apache.org/).

While I'm generally a massive fan of AWS and similar cloud computing services,
and use them extensively in my day job, the idea of Kaggle entering a phase
where these services are essential in order to be competitive worries me. It
certainly makes these competitions hosted by big corporations feel more
exploitative and I would like to see the prize money increase dramatically to
help address that.

In the mean time, I'm going to take the opportunity to demonstrate how a data
set of this size can be analysed on a modestly resourced PC using R with some
clever coding. In order to facilitate the more frequent posting that I'm
aspiring to, this particular post is only going to go as far as calculating a
summary statistic on the entire training data set and make a first simple
submission the basis of that. A second post will (hopefully!) soon follow this
one with some more interesting modelling techniques, again driven by hardware
limitations.

Normally when you read a data set into R you might do something like:

{% highlight r %}
X <- read.csv("train_numeric.csv")
{% endhighlight %}

You might be aware that R can handle reading compressed files on the fly, for
example if you have a gzipped file you can do:

{% highlight r %}
X <- read.csv("train_numeric.csv.gz")
{% endhighlight %}

It's that simple - R will work out that it's a gzipped file and handle the
decompression.

Handling zip files - which the data for this competition are stored in - is a
bit less elegant due to the fact that zip files can and often do contain
multiple files:

{% highlight r %}
X <- read.csv(unz("train_numeric.csv.zip","train_numeric.csv"))
{% endhighlight %}

Notice how it's necessary to explicitly use the `unz` function. For a gzipped
file, R is actually using the `gzfile` function behind the scenes but `unz` is
a bit more complicated because it takes a second argument which specifies which
file _within_ the zip file you want to read.

If you've downloaded the data for this competition I don't recommend running the
above command! This will read the entire numeric training data set into memory
which is exactly what we're trying to avoid.

Instead let's do the following. First, open a connection to the zip file:

{% highlight r %}
conn <- unz("train_numeric.csv.zip","train_numeric.csv","r")
{% endhighlight %}

We're now using a third argument to the function which specifies how we want to
open the file (e.g. for reading or writing). In this case we use `"r"` denoting
that we want to read the file.

Next let's read the first line of the file - which contains the column names -
and save them for future use:

{% highlight r %}
columns <- scan(conn,what=character(0),sep=",",nlines=1)
{% endhighlight %}

Here we use the `scan` function, we specify `nlines=1` so we only read the
first line, we set `what=character(0)` to denote that we're expecting text data
and finally we set `sep=","` to denote that the data is separated by commas.

It's important at this stage to understand what's going on. We've got the zip
file open and we can read data from it by passing the connection `conn` to
`scan`. We've read one line from the file, so we've got some data from it into
memory but only a very modest amount. However because the connection is still
open, if we read another line (or more) from the file, we'll start from the
_second line_. In this way, we can process each line of the file individually.
In fact, we can process a few lines in one go in order to be a bit more
efficient but either way we've now achieved a set up where the data can be read
and manipulated without reading the entire file into memory.

Here's a function for reading some lines from the file into a data frame:

{% highlight r %}
read_chunk <- function(chunk_size=50)
{
  raw <- scan(conn,
              what=numeric(0),
              sep=",",
              nlines=chunk_size,
              quiet=TRUE)
  
  chunk <- as.data.frame(t(matrix(raw,nrow=length(columns))))
  names(chunk) <- columns
  return(chunk)
}
{% endhighlight %}

It takes a single argument `chunk_size` which specifies how many lines to read.
It then reads them using the `scan` function as before, except this time we
know the data is going to be `numeric` rather than text and we set `quiet=TRUE`
because we're going to end up calling this function lots of times and we don't
want it to print out how many elements it's read, which is the default behaviour
of `scan`.

After reading the lines into what will be a long vector of numbers, we turn it
into a data frame by first turning it into a matrix. We use `t` to transpose
the matrix as reading the data line by line from the file means it's in
row-major rather than column-major format. We then assign the column names which
we previously stored to the data frame and return it.

Now let's use this functionality to do some analysis. Here's a simple function
which makes use of `read_chunk` to process the file in chunks and stores the
entire response vector, throwing away all of the features:

{% highlight r %}
read_response_vector <- function()
{
  response_vector <- numeric(0)
  chunk_size <- 50
  rows_read <- 50
  while(rows_read == chunk_size)
  {
    chunk <- read_chunk(chunk_size=chunk_size)
    rows_read <- nrow(chunk)
    response_vector <- c(response_vector,chunk$Response)
    if(length(response_vector) %% 1000 == 0)
    {
      print(paste("Read",length(response_vector),"lines"))
    }
  }
  return(response_vector)
}
{% endhighlight %}

It repeatedly calls `read_chunk` with a `chunk_size` of 50 lines, until it hits
a call where less than 50 lines were read which will denote the entire file
having been read. For each chunk it appends the `Response` field to a growing
`response_vector` which is returned when reading the file is complete. Some
basic progress reporting is included so that every `1000` lines an informative
message will be printed.

After calling this function it's a good idea to close the connection:

{% highlight r %}
close(conn)
{% endhighlight %}

Now we can use a trivial analysis of the response to make a first submission. By
using the table function you can see that there are 6,879 positive cases and
1,176,868 negative cases or equivalently that the proportion of positive cases
is 0.005811208. So a simple submission would be to randomly pick 0.58% of the
test samples to predict as positive:

{% highlight r %}
submission <- read.csv(unz("sample_submission.csv.zip","sample_submission.csv"),
                       header=TRUE,
                       stringsAsFactors=FALSE)

number_of_positives <- floor(0.005811208 * nrow(submission))
# For reproducibility
set.seed(123)
indices <- sample(1:nrow(submission),replace=FALSE,size=number_of_positives)
submission$Response[indices] <- 1
write.csv(submission,gzfile("submission001.gz"),row.names=FALSE)
{% endhighlight %}

Note again how we use `unz` in `read.csv` to read the sample submission file
without first decompressing it. Also note that the submission file is written
directly as a gzipped file by using the `gzfile` function in `write.csv`. This
submission should score you 0.00122 which is far from competitive but will at
least beat the benchmark!

I hope you've found this post informative. As always, please feel free to
contact me via any of the methods on the [About page]({{ site.baseurl }}about)
or leave your comments below. Look out for a follow up post in the next week or
so building upon this method of chunked processing to carry out some more
interesting analysis of the data set.
