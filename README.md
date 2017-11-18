# Meta-Learning approach for automatic kernel selection for support vector machines

Author:

- [Michael M.Meskhi](michaelmm.com)

This project is based on this paper by [Shawkat Ali and Kate A. Smith-Miles]().
By collection meta-features about a task at hand (dataset), we generate rules based on collected statistical information and perform automatic kernel selection for our Support Vector Machine.

Meta-features:

**1- Rule for polynomial kernel:**

```Python
if md <= 218.276:
	kernel = 'poly'
```

- *Mahanlanobis distance (md)*

<img src="http://bit.ly/2mHLiCD" align="center" border="0" alt="md_{rs}^{2}=(x_{r}-x_{s})V^{-1}(x_{r}-x_{s})^{\prime}" width="242" height="26" />

**2- Rule for rbf kernel:**

```Python
elif (R > 9 and p-norm_cdf > 7.2957) or (p-du_cdf <= 2.8185):
	kernel = 'rbf'
```

- *Range (R)*

	<img src="http://bit.ly/2mEFIAQ" align="center" border="0" alt="R = max(X) - min(X)" width="181" height="18" />

- *Normal cdf (P-norm_cdf)*

	<img src="http://bit.ly/2mDKhv9" align="center" border="0" alt="P_{norm\_cdf} = F(x|\mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\int^{x}_{-\infty}e^{-(x-\mu)^{2}/2\sigma^{2}}dt" width="394" height="49" />

- *Discrete uniform cdf*

	<img src="https://latex.codecogs.com/gif.latex?P_{du\_cdf} = F(x|N) = \frac{\lfloor(x)\rfloor}{N}I_{(0,1...,n)}(x)" />

**3- Rule for Laplace kernel:**

```Python
elif y-gama_pdf <= 17.1671:
	kernel = 'laplace'
```

- *Gamma pdf*

	<img src="https://latex.codecogs.com/gif.latex?y_{gamma\_pdf} = f)(x|a, b) = \frac{1}{b^{a}\Gamma(a)}x^{a-1}e^{x/b}" /> 
	
**4- Rule for spline kernel:**

```Python
elif (y-ray_pdf > 20.2875) or (M <= 90.7233):
	kernel = 'spline'
```

- *Median (M)*

	<img src="https://latex.codecogs.com/gif.latex?M(a) = \frac{a_{\lfloor|x|/2\rfloor}+a_{\lfloor|x|/2+0.5\rfloor}}{2}" /> 
	
- *Rayleigh pdf*

	<img src="https://latex.codecogs.com/gif.latex?y_{ray\_pdf} = f(x|v) = \frac{\Gamma((v+1)/2)}{\Gamma(v/2)}\frac{1}{\sqrt{v\pi}}\frac{1}{(1+(x^{2}/v))^{(v+1)/2}}" /> 	
**5- Default rule if any of above are FALSE:**

```Python
else:
	kernel = 'rbf'
```