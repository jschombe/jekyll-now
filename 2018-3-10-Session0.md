
![alt text](http://datascience.uci.edu/wp-content/uploads/sites/2/2014/09/data_science_logo_with_image1.png 'UCI_data_science')

## Preliminaries

1.  If you haven't do so already, download and **install the [Anaconda Scientific Python Distribution version 2.7](https://store.continuum.io/cshop/anaconda/)**.  If it offers to make itself your default Python distribution, allow it.
1. Whether you've just installed Anaconda, or you have done so previously, you should now **update Anaconda** to the latest version of the distribution.  It changes a lot so do this today even if you did recently.
 1. Open a terminal or command prompt.
 1. Type ```conda update conda``` and press enter or return.  Confirm that you'd like it to make any changes that it offers.
 1. Type ```conda update anaconda``` and press enter or return.  Confirm that you'd like it to make any changes that it offers.
1. **Download the code repository**.  
 1. Go to [bit.ly/uci_predictive](http://bit.ly/uci_predictive) and click the "download zip" button on the right to download a zip file containing this entire repository.
 1. Unzip that file into a directory you know how to find; you'll need it several times throughout the day.  
1. **Start an ipython notebook server**.
 1. Open a terminal and type ```ipython notebook```.  Navigate to the directory where you unzipped this repository.
 1. Open "Test Notebook.ipynb".
 1. Click "Cell" at the top of the opened notebook, followed by "Run All" and ensure that 1) there are no errors and that 2) the output from the first cell is the same as that in the second.  If it doesn't match, raise your hand.
 1. If everything looks good, close the browser tab containing the test notebook but keep open the tab listing all the other notebooks.

### Schedule for Today

|Start Time | Session |
|-----------|---------|
|8:30am     | Check In|
|9:00am     | **The IPython Notebook and Pandas** |
|10:30am    | Coffee & Bagels|
|10:45am    | **Linear Regression and Predictive Modeling** |
|12:30pm    | Lunch|
|1:00pm     | **Out of Sample Prediction** |
|2:45pm     | Afternoon break|
|3:00pm     | **Logistic Regression** |

## Predictive Modeling with Python - _IPython Notebooks and Viewing Data in Python_
#### Author: Kevin Bache

## Outline
1. IPython and IPython Notebooks
1. Numpy
1. Pandas

## Python and IPython
* `python` is a programming language and also the name of the program that runs scripts written in that language.
* If you're running scripts from the command line you can use either `ipython` with something like `ipython my_script.py` or `python` with something like `python my_script.py`
* If you're using the command line interpreter interactively to load and explore data, try out a new package, etc. always use `ipython` over `python`.  This is because `ipython` has a bunch of features like tab completion, inline help, and easy access to shell commands which are just plain great (more on these in a bit).

## IPython Notebook
* IPython notebook is an interactive front-end to ipython which lets you combine snippets of python code with explanations, images, videos, whatever.  
* It's also really convenient for conveying experimental results.
* http://nbviewer.ipython.org

### <span style="color:red">Self-Driven IPython Notebook Exercise #1</span>
1. Start a terminal window and cd to the directory where you stored the course files
1. Start the IPython Notebook server with the command `ipython notebook`.  The IPython notebook server runs your python code behind the scenes and renders the output into the notebook
1. Create a new notebook by clicking New (top right) >> Python 2 Notebook

### Notebook Concepts
* **Cells** -- That grey box is called a cell.  An IPython notebook is nothing but a series of cells.  
* **Selecting** -- You can tell if you have a cell selected because it will have a thin, black box around it.
* **Running a Cell** -- Running a cell displays its output.  You can run a cell by pressing **`shift + enter`** while it's selected (or click the play button toward the top of the screen). 
* **Modes** -- There are two different ways of having a cell selected:
  * **Command Mode** -- Lets you delete a cell and change its type (more on this in a second).
  * **Edit Mode** -- Lets you change the contents of a cell.

### Aside: Keyboard Shortcuts That I Use A Lot
* (When describing keyboard shortcuts, `+` means 'press at the same time', `,` means 'press after'
* **`Enter`** -- Run this cell and make a new one after it
* **`Esc`** -- Stop editing this cell
* **`Option + Enter`** -- Run this cell and make a new cell after it (Note: this is OSX specific.  Check help >> keyboard shortcuts to find your operating system's version)
* **`Shift + Enter`** -- Run this cell and don't make a new one after it
* **`Up Arrow`** and **`Down Arrow`**  -- Navigate between cells (must be in command mode)
* **`Esc, m, Enter`** -- Convert the current cell to markdown and start editing it again
* **`Esc, y, Enter`** -- Convert the current cell to a code cell and start editing it again
* **`Esc, d, d`** -- Delete the current cell
* **`Esc, a`** -- Create a new cell above the current one
* **`Esc, b`** -- Create a new cell below the current one
* **`Command + /`** -- Toggle comments in Python code (OSX)
* **`Ctrl + /`** -- Toggle comments in Python code (Linux / Windows)

### <span style="color:red">Self-Driven IPython Notebook Exercise #2</span>
1. Click Help >> User Interface Tour and take the tour
1. Click Help >> Keyboard Shortcuts.  Mice are for suckers.

### More Notebook Concepts
* **Cell Types** -- There are 3 types of cells: python, markdown, and raw.
  * **Python Cells** -- Contain python code. Running a python cell displays its output. Press **`y`** in command mode convert any selected cell into a python cell. All cells start their lives as python cells.
  * **Markdown Cells** -- Contain formatted text, lists, links, etc. Press **`m`** in command mode to convert the selected cell into a markdown cell.
  * **Raw Cells** -- Useful for a few advanced corner cases.  We won't deal with these at all today.
  
### <span style="color:red">Self-Driven IPython Notebook Exercise #3</span>:
1. Partner up with someone next to you
  1. Create a code cell
    1. IPython notebooks offer tab completion.  Try typing: ```a<tab>``` 
    1. IPython notebooks offer inline help.  Try typing: ```?abs``` and then running the cell
  1. Create and render a markdown cell which contains 
    1. Bold text
    1. A nested numbered list
    1. A working link to UCI's website
    1. An image
    1. Some rendered LaTex. 
  1. **Hint**: The following websites might be useful:
    1. Check out this [Markdown Cheat Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).  Markdown is a set of simple commands for formatting text to make it pretty.  It isn't specific to IPython Notebooks; it's used all over (for example, a lot of blogging platforms let you write your content in markdown because it's easy and HTML is a pain in the butt).
    1. Check out this [stackoverflow post](http://stackoverflow.com/questions/13208286/how-to-write-latex-in-ipython-notebook) about using LaTex in IPython Notebooks.

## Numpy
Numpy is the main package that you'll use for doing scientific computing in Python.  Numpy provides a multidimensional array datatype called `ndarray` which can do things like vector and matrix computations.

### Resources:
* [Official Numpy Tutorial](http://wiki.scipy.org/Tentative_NumPy_Tutorial)
* [Numpy, R, Matlab Cheat Sheet](http://mathesaurus.sourceforge.net/matlab-python-xref.pdf)
* [Another Numpy, R, Matlab Cheat Sheet](http://sebastianraschka.com/Articles/2014_matrix_cheatsheet_table.html)


```python
# you don't have to rename numpy to np but it's customary to do so
import numpy as np

# you can create a 1-d array with a list of numbers
a = np.array([1, 4, 6])
print 'a:'
print a
print 'a.shape:', a.shape
print 

# you can create a 2-d array with a list of lists of numbers
b = np.array([[6, 7], [3, 1], [4, 0]])
print 'b:'
print b
print 'b.shape:', b.shape
print
```

    a:
    [1 4 6]
    a.shape: (3L,)
    
    b:
    [[6 7]
     [3 1]
     [4 0]]
    b.shape: (3L, 2L)
    
    


```python
# you can create an array of ones
print 'np.ones(3, 4):'
print np.ones((3, 4))
print

# you can create an array of zeros
print 'np.zeros(2, 5):'
print np.zeros((2, 5))
print

# you can create an array which of a range of numbers and reshape it
print 'np.arange(6):'
print np.arange(6)
print 
print 'np.arange(6).reshape(2, 3):'
print np.arange(6).reshape(2, 3)
print

# you can take the transpose of a matrix with .transpose or .T
print 'b and b.T:'
print b
print 
print b.T
print 
```

    np.ones(3, 4):
    [[ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]]
    
    np.zeros(2, 5):
    [[ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    
    np.arange(6):
    [0 1 2 3 4 5]
    
    np.arange(6).reshape(2, 3):
    [[0 1 2]
     [3 4 5]]
    
    b and b.T:
    [[6 7]
     [3 1]
     [4 0]]
    
    [[6 3 4]
     [7 1 0]]
    
    


```python
# you can iterate over rows
i = 0
for this_row in b:
    print 'row', i, ': ', this_row
    i += 1 
print 
    
# you can access sections of an array with slices
print 'first two rows of the first column of b:'
print b[:2, 0]
print
```

    row 0 :  [6 7]
    row 1 :  [3 1]
    row 2 :  [4 0]
    
    first two rows of the first column of b:
    [6 3]
    
    


```python
# you can concatenate arrays in various ways:
print 'np.hstack([b, b]):'
print np.hstack([b, b])
print

print 'np.vstack([b, b]):'
print np.vstack([b, b])
print
```

    np.hstack([b, b]):
    [[6 7 6 7]
     [3 1 3 1]
     [4 0 4 0]]
    
    np.vstack([b, b]):
    [[6 7]
     [3 1]
     [4 0]
     [6 7]
     [3 1]
     [4 0]]
    
    


```python
# note that you get an error if you pass in print 'np.hstack(b, b):'
print np.hstack(b, b)
print

```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-7-be873fe9b3f9> in <module>()
          1 # note that you get an error if you pass in print 'np.hstack(b, b):'
    ----> 2 np.hstack(b, b)
          3 print
    

    TypeError: hstack() takes exactly 1 argument (2 given)



```python
# you can perform matrix multiplication with np.dot()
c = np.dot(a, b)
print 'c = np.dot(a, b):'
print c
print

# if a is already a numpy array, then you can also use this chained 
# matrix multiplication notation.  use whichever looks cleaner in 
# context
print 'a.dot(b):'
print a.dot(b)
print


# you can perform element-wise multiplication with * 
d = b * b
print 'd = b * b:'
print d
print

a.dot(b)
```

    c = np.dot(a, b):
    [42 11]
    
    a.dot(b):
    [42 11]
    
    d = b * b:
    [[36 49]
     [ 9  1]
     [16  0]]
    
    




    array([42, 11])



### Arrays and Matrices
In addition to arrays which can have any number of dimensions, Numpy also has a `matrix` data type which always has exactly 2.  **DO NOT USE `matrix`**.  

The original intention behind this data type was to make Numpy feel a bit more like Matlab, mainly by making the `*` operator perform matrix multiplication so you don't have to use `np.dot`.  But `matrix` isn't as well developed by the Numpy people as `array` is.  `matrix` is slower and using it will sometimes throw errors in other people's code because everyone expects you to use `array`.


```python
# you can convert a 1-d array to a 2-d array with np.newaxis
print 'a:'
print a
print 'a.shape:', a.shape
print 
print 'a[np.newaxis] is a 2-d row vector:'
print a[np.newaxis]
print 'a[np.newaxis].shape:', a[np.newaxis].shape
print

print 'a[np.newaxis].T: is a 2-d column vector:'
print a[np.newaxis].T
print 'a[np.newaxis].T.shape:', a[np.newaxis].T.shape
print

```

    a:
    [1 4 6]
    a.shape: (3L,)
    
    a[np.newaxis] is a 2-d row vector:
    [[1 4 6]]
    a[np.newaxis].shape: (1L, 3L)
    
    a[np.newaxis].T: is a 2-d column vector:
    [[1]
     [4]
     [6]]
    a[np.newaxis].T.shape: (3L, 1L)
    
    


```python
# numpy provides a ton of other functions for working with matrices
m = np.array([[1, 2],[3, 4]])
m_inverse = np.linalg.inv(m)
print 'inverse of [[1, 2],[3, 4]]:'
print m_inverse
print

print 'm.dot(m_inverse):'
print m.dot(m_inverse)
```

    inverse of [[1, 2],[3, 4]]:
    [[-2.   1. ]
     [ 1.5 -0.5]]
    
    m.dot(m_inverse):
    [[  1.00000000e+00   1.11022302e-16]
     [  0.00000000e+00   1.00000000e+00]]
    


```python
# and for doing all kinds of sciency type stuff.  like generating random numbers:
np.random.seed(5678)
n = np.random.randn(3, 4)
print 'a matrix with random entries drawn from a Normal(0, 1) distribution:'
print n
```

    a matrix with random entries drawn from a Normal(0, 1) distribution:
    [[-0.70978938 -0.01719118  0.31941137 -2.26533107]
     [-1.37745366  1.94998073 -0.56381007 -0.84373759]
     [ 0.22453858 -0.39137772  0.60550347 -0.68615034]]
    

### <span style="color:red">Self-Driven Numpy Exercise</span>
1. In the cell below, add a column of ones to the matrix `X_no_constant`.  This is a common task in linear regression and general linear modeling and something that you'll have to be able to do later today.  
1. Multiply your new matrix by the `betas` vector below to make a vector called `y`
1. You'll know you've got it when the cell prints '\*\*\*\*\*\* Tests passed! \*\*\*\*\*\*' at the bottom.

Specificically, given a matrix:

\begin{equation*}
\qquad
\mathbf{X_{NoConstant}} = 
\left( \begin{array}{ccc}
x_{1,1} & x_{1,2} & \dots & x_{1,D} \\
x_{2,1} & x_{2,2} & \dots & x_{2,D} \\
\vdots & \vdots & \ddots & \vdots \\
x_{i,1} & x_{i,2} & \dots & x_{i,D} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N,1} & x_{N,2} & \dots & x_{N,D} \\
\end{array} \right)
\qquad
\end{equation*}

We want to convert it to:
\begin{equation*}
\qquad
\mathbf{X} = 
\left( \begin{array}{ccc}
1 & x_{1,1} & x_{1,2} & \dots & x_{1,D} \\
1 & x_{2,1} & x_{2,2} & \dots & x_{2,D} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{i,1} & x_{i,2} & \dots & x_{i,D} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{N,1} & x_{N,2} & \dots & x_{N,D} \\
\end{array} \right)
\qquad
\end{equation*}

So that if we have a vector of regression coefficients like this:

\begin{equation*}
\qquad
\beta = \left( \begin{array}{ccc}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_j \\
\vdots \\
\beta_D
\end{array} \right)
\end{equation*}

We can do this:

\begin{equation*}
\mathbf{y} \equiv \mathbf{X} \mathbf{\beta} 
\end{equation*}



```python
np.random.seed(3333)
n_data = 10 # number of data points. i.e. N
n_dim = 5   # number of dimensions of each datapoint.  i.e. D

betas = np.random.randn(n_dim + 1)

X_no_constant = np.random.randn(n_data, n_dim)
print 'X_no_constant:'
print X_no_constant
print 

# INSERT YOUR CODE HERE!

# Tests:
y_expected = np.array([-0.41518357, -9.34696153, 5.08980544, 
                       -0.26983873, -1.47667864, 1.96580794, 
                       6.87009791, -2.07784135, -0.7726816, 
                       -2.74954984])
np.testing.assert_allclose(y, y_expected)
print '****** Tests passed! ******'
```

    X_no_constant:
    [[-0.92232935  0.27352359 -0.86339625  1.43766044 -1.71379871]
     [ 0.179322   -0.89138595  2.13005603  0.51898975 -0.41875106]
     [ 0.34010119 -1.07736609 -1.02314142 -1.02518535  0.40972072]
     [ 1.18883814  1.01044759  0.3108216  -1.17868611 -0.49526331]
     [-1.50248369 -0.196458    0.34752922 -0.79200465 -0.31534705]
     [ 1.73245191 -1.42793626 -0.94376587  0.86823495 -0.95946769]
     [-1.07074604 -0.06555247 -2.17689578  1.58538804  1.81492637]
     [-0.73706088  0.77546031  0.42653908 -0.51853723 -0.53045538]
     [ 1.09620536 -0.69557321  0.03080082  0.25219596 -0.35304303]
     [-0.93971165  0.04448078  0.04273069  0.4961477  -1.7673568 ]]
    
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-12-c4d2ae827182> in <module>()
         17                        6.87009791, -2.07784135, -0.7726816,
         18                        -2.74954984])
    ---> 19 np.testing.assert_allclose(y, y_expected)
         20 print '****** Tests passed! ******'
    

    NameError: name 'y' is not defined


## Pandas
Pandas is a python package which adds some useful data analysis features to numpy arrays.  Most importantly, it contains a `DataFrame` data type like the r `dataframe`: a set of named columns organized into something like a 2d array.  Pandas is great.

### Resources:
* [10 Minutes to Pandas](http://pandas.pydata.org/pandas-docs/dev/10min.html)
* [Pandas Data Structures Tutorial](http://pandas.pydata.org/pandas-docs/stable/dsintro.html)
* [Merge, Join, Concatenate Tutorial](http://pandas.pydata.org/pandas-docs/dev/merging.html)
* [Another Numpy/Pandas Tutorial](https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii)


```python
# like with numpy, you don't have to rename pandas to pd, but it's customary to do so
import pandas as pd

b = np.array([[6, 7], [3, 1], [4, 0]])
df = pd.DataFrame(data=b,  columns=['Weight', 'Height'])
print 'b:'
print b
print 
print 'DataFame version of b:'
print df
print
```


```python
# Pandas can save and load CSV files.  
# Python can do this too, but with Pandas, you get a DataFrame 
# at the end which understands things like column headings
baseball = pd.read_csv('data/baseball.dat.txt')

# A Dataframe's .head() method shows its first 5 rows
baseball.head()
```


```python
# you can see all the column names
print 'baseball.keys():'
print baseball.keys()
print

print 'baseball.Salary:'
print baseball.Salary
print 
print "baseball['Salary']:"
print baseball['Salary']
```


```python
baseball.info()
```


```python
baseball.describe()
```


```python
baseball
```


```python
# You can perform queries on your data frame.  
# This statement gives you a True/False vector telling you 
# whether the player in each row has a salary over $1 Million
millionaire_indices = baseball['Salary'] > 1000
print millionaire_indices
```


```python
# you can use the query indices to look at a subset of your original dataframe
print 'baseball.shape:', baseball.shape
print "baseball[millionaire_indices].shape:", baseball[millionaire_indices].shape
```


```python
# you can look at a subset of rows and columns at the same time
print "baseball[millionaire_indices][['Salary', 'AVG', 'Runs', 'Name']]:"
baseball[millionaire_indices][['Salary', 'AVG', 'Runs', 'Name']]
```

## Pandas Joins - If you have time
The real magic with a Pandas DataFrame comes from the merge method which can match up the rows and columns from two DataFrames and combine their data.  Let's load another file which has shoesize for just a few players


```python
# load shoe size data
shoe_size_df = pd.read_csv('data/baseball2.dat.txt')
shoe_size_df
```


```python
merged = pd.merge(baseball, shoe_size_df, on=['Name'])
merged
```


```python
merged_outer = pd.merge(baseball, shoe_size_df, on=['Name'], how='outer')
merged_outer.head()
```

### <span style="color:red">Self-Driven Pandas Exercise</span>
1. Partner up with someone next to you.  Then, on one of your computers:
  1. Prepend a column of ones to the dataframe `X_df` below.  Name the new column 'const'.
  1. Again, matrix multiply `X_df` by the `betas` vector and assign the result to an new variable: `y_new`
  1. You'll know you've got it when the cell prints '\*\*\*\*\*\* Tests passed! \*\*\*\*\*\*' at the bottom.

  **Hint**: This stackoverflow post may be useful: http://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns


```python
np.random.seed(3333)
n_data = 10 # number of data points. i.e. N
n_dim = 5   # number of dimensions of each datapoint.  i.e. D

betas = np.random.randn(n_dim + 1)

X_df = pd.DataFrame(data=np.random.randn(n_data, n_dim))

# INSERT YOUR CODE HERE!

# Tests:
assert 'const' in X_df.keys(), 'The new column must be called "const"'
assert np.all(X_df.shape == (n_data, n_dim+1))
assert len(y_new == n_data)
print '****** Tests passed! ******'
```
