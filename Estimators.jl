# This script lists classical estimators used to estimate linear models, and estimators using kernels ---------------------------

# OLS function. Include a vector of ones in x if one wants to include an intercept
function ols(x, y)
     beta =inv(x'*x)* x'*y
	 return(beta)
end

# The Two Stage Least Squares (TSLS) function. Include a vector of ones in x and/or z if one wants to include an intercept the first and/or second stage
function tsls(x, y, z)
    z = hcat(ones(size(z, 1), 1), z)
    pz = z*inv(z'*z)*z'
    beta = inv(x'*pz*x)* x'*pz*y
	 return(beta)
end

# Function that considers a kernel regression estimator of the conditional expectation of x given z, and uses the fitted values in the second stage equation.
# It uses the KernelEstimator.jl package, found on https://github.com/panlanfeng/KernelEstimator.jl
# Does not include a ridging option for small bandwidths, so NAs can be returned.
# Here, the bandwidth is automatically chosen via Cross Validation
using KernelEstimator
function np_tsls(x, y, z; option = localconstant)
     xhat = npr(x, z, reg = option, kernel = gaussiankernel)
     beta = inv(xhat'*x)* xhat'*y
	 return(beta)
end

# Here, the bandwidth is fixed by the user.
#Pkg.add("KernelEstimator")
using KernelEstimator
function np_tsls_h(x, y, z, h; option = localconstant)
     xhat = npr(x, z, reg = option, kernel = gaussiankernel, h = h)
     beta = inv(xhat'*x)* xhat'*y
	 return(beta)
end


# The k-class estimator of Theil(1958). It includes the OLS (k = 0) and TSLS (k = 1) as special cases
function kclass(x, y, z; k = 1)
    n = size(y, 1)
    Pz = z*inv(z'*z)*z'
    Mz = Matrix{Float64}(I, n, n) - Pz
    betahat = inv(x'*(Matrix{Float64}(I, n, n) - k*Mz)*x)*x'*(Matrix{Float64}(I, n, n) - k*Mz)*y
    return betahat
end

# The Limited Information Maximum Likelihood (LIML) estimator, also a special case of the k-class estimator.
function liml(x, y, z)
    n = size(y, 1)
    Pz = z*inv(z'*z)*z'
    Mz = Matrix{Float64}(I, n, n) - Pz
    w = hcat(y, x)
    function obj(k)
        expr = det(w'*w - k.*w'*Mz*w)
    end
    kliml = nlsolve(obj, vec([0.5])).zero
    betahat = inv(x'*(Matrix{Float64}(I, n, n) - kliml.*Mz)*x)*x'*(Matrix{Float64}(I, n, n) - kliml.*Mz)*y
    return betahat
end

# The WMDF estimator of Antoine & Lavergne (2014)
function wmdf(x, y, z; intercept = "false")
    n = size(y, 1)
     if intercept == "true"
       Y1star = hcat(ones(n, 1), x)
     else Y1star = x
     end
      Ystar = hcat( y , Y1star )
      z = z./repeat(std(z, dims = 1), n, 1)  # some rescaling

    if size(z, 2)>1
      Ktilde = ones(n, n)
      for i = 1:size(z, 2)
        K = dnorm( repeat(z[:, i], 1, n) -  repeat(z[:, i]', n, 1) )
        K = K - Diagonal(K)
        Ktilde = Ktilde.*K
      end
     else
        Ktilde = dnorm( repeat(z, 1, n) -  repeat(z', n, 1) )
        Ktilde = Ktilde - Diagonal(Ktilde)
    end

      eig = eigvals(inv(Ystar'*Ystar)*(Ystar'*Ktilde*Ystar))
      lambdatilde = minimum(eig)

      lambda_WMDF = (lambdatilde - (1 - lambdatilde)/n)/(1 - (1 - lambdatilde)/n)
      In = Diagonal(vec(ones(n, 1)))
      WMD = inv(Y1star'*(Ktilde-lambdatilde.*In)*Y1star)*(Y1star'*(Ktilde-lambdatilde*In)*y)
      WMDF = inv(Y1star'*(Ktilde-lambda_WMDF.*In)*Y1star)*(Y1star'*(Ktilde-lambda_WMDF*In)*y)
      return (WMD = WMD, WMDF = WMDF)
end

# Example
#=
 include("auxiliary functions.jl")
 n = 1000
 mu = [0.0, 0]
 sigma = [1.0, 0.5, 0.5, 1.0]
 sigma = reshape(sigma, 2, 2)
 # Generate some data where the coeffecieint of interest, beta_0, is equal to 1.
 data = dgp(n, 1, mu, sigma; design = "linear")

 OLS = ols(data.x, data.y)
 TSLS = tsls(data.x, data.y, data.z)
 LIML = liml(data.x, data.y, data.z)
 KCLASS_0 = kclass(data.x, data.y, data.z, k = 0)    # returns the same as OLS
 KCLASS_1 = kclass(data.x, data.y, data.z, k = 1)    # returns the same as TSLS
 WMDF = wmdf(data.x, data.y, data.z, intercept = "false")
 NP = np_tsls(data.x, data.y, data.z, option = localconstant)
=#


# Estimators computing or using kernels -------------------------------------

# This function computes the Nadaraya-Watson (type = "lc") and local linear (type = "ll") kernel regression estimators
# A Gaussian kernel is used
# No ridging to handle small bandwidths, instead a numerical adjustment to avoid dividing by 0. It is not ideal.
function lcll( dep, expl; bw = 0.1, type = "lc" )
  n = size(expl, 1)
  firsterm = repeat(expl, 1, n)
  secterm = firsterm'
  U = (firsterm-secterm)./bw
  K = gaussian(U)   # Replace that by any other kernel function if desired
  if type == "ll"

    Sn1 = sum(K.*(firsterm-secterm), dims = 2)
    Sn1 = repeat(Sn1, 1, n)

    Sn2 = sum(K.*(firsterm-secterm).^2, dims = 2)
    Sn2 = repeat(Sn2, 1, n)

    B = K.*(Sn2 - (firsterm-secterm).*Sn1)
    SB = sum(B, dims = 2)

  if any(SB==0)
      println("Careful mate ! An adjustment was made to avoid Nan!")
      ind = find( x -> x == 0, SB)
      SB[which(SB==0)]= 0.0001
  end  # if there is a sum that is equal to 0}

    L = B./repeat(SB, 1, n)

 else
    SB = sum(K, dims = 2)
    if any(SB==0)
        println("Careful mate ! An adjustment was made to avoid Nan!")
        ind = find( x -> x == 0, SB)
        SB[which(SB==0)]= 0.0001
    end  # if there is a sum that is equal to 0}

    L = K./repeat(SB, 1, n)
 end
  fit = L*dep
  return(fit)
end


# Function that considers a kernel regression estimator of the conditional expectation of x given z, and uses the fitted values in the second stage equation.
# It uses the lcll function, present in this script.
# Does not include a ridging option for small bandwidths, so a numerical adjustment is made to avoid dividing by 0. But it is not ideal.
function kernel_tv(x, y, z; option, bw )
    if option != "lc" && option != "ll"
    error("option should be lc or ll ! Careful mate !")
    end
     xhat = lcll(x, z, bw = bw, type = option)
     beta = inv(xhat'*x)* xhat'*y
	 return(beta)
end

# Example
#=
include("auxiliary function.jl")
 n = 1000
 mu = [0.0, 0]
 sigma = [1.0, 0.5, 0.5, 1.0]
 sigma = reshape(sigma, 2, 2)
 data = dgp(n, 1, mu, sigma)

 OLS = ols(data.x, data.y)
 TSLS = tsls(data.x, data.y, data.z)
 LIML = liml(data.x, data.y, data.z)
 KCLASS_0 = kclass(data.x, data.y, data.z, k = 0)
 KCLASS_1 = kclass(data.x, data.y, data.z, k = 1)
 WMDF = wmdf(data.x, data.y, data.z, intercept = "true")
 NP = np_tsls(data.x, data.y, data.z, option = localconstant)
 TV = kernel_tv(data.x, data.y, data.z; option = "ll", bw = 1)
 # Note that if the bandwidth is very high, the estimator coincides with teh TSLS one
 TV_500 = kernel_tv(data.x, data.y, data.z; option = "ll", bw = 500)
=#


# Extract the "hat" matrix when using the Nadaraya-Watson or local linear kernel estimators.
 function kernel_hat_matrix( dep, expl; bw = 0.1, type = "lc" )
      n = size(expl, 1)
      firsterm = repeat(expl, 1, n)
      secterm = firsterm'
      U = (firsterm-secterm)./bw
      K = gaussian(U)
      if type == "ll"

        Sn1 = sum(K.*(firsterm-secterm), dims = 2)
        Sn1 = repeat(Sn1, 1, n)

        Sn2 = sum(K.*(firsterm-secterm).^2, dims = 2)
        Sn2 = repeat(Sn2, 1, n)

        B = K.*(Sn2 - (firsterm-secterm).*Sn1)
        SB = sum(B, dims = 2)

      if any(SB==0)
          println("Careful mate ! An adjustment was made to avoid Nan!")
          ind = find( x -> x == 0, SB)
          SB[which(SB==0)]= 0.0001
      end  # if there is a sum that is equal to 0}
        L = B./repeat(SB, 1, n)
     else
        SB = sum(K, dims = 2)
        if any(SB==0)
            println("Careful mate ! An adjustment was made to avoid Nan!")
            ind = find( x -> x == 0, SB)
            SB[which(SB==0)]= 0.0001
        end  # if there is a sum that is equal to 0}
        L = K./repeat(SB, 1, n)
     end
  return(L)
end

# Implements the AIC criterion function for the bandwidth from Hurvic, Simonoff & Tsai (1999)
 function hurvic_AIC(dep, expl; bw = 0.1, type = "lc")
  H = kernel_hat_matrix(dep, expl, bw = bw, type = type)
  n = size(dep, 1)
  sig2 = ( dep'*( Matrix{Float64}(I, n, n) - H )'*( Matrix{Float64}(I, n, n) - H )'*dep ) / n
  crit = log(sig2) + (1 +  tr(H)/ n)/( 1 - (tr(H) + 2) / n)
  return(crit)
end

# An extension of the AIC function of Hurvic, Simonoff & Tsai (1999) for the IV case
function iv_AIC(expl, endo, instru; bw = 0.1, type = "lc")
  n = size(expl, 1)
  H = kernel_hat_matrix(endo, instru, bw = bw, type = type)
  # In the IV case, the effective hat matrix is (X'H'X)^(-1) X'H'
  H = endo*inv(endo'*H'*endo)*endo'*H'
  sig2 = ( expl'*( Matrix{Float64}(I, n, n) - H )'*( Matrix{Float64}(I, n, n) - H )*expl ) / n
  crit = log(sig2) + (1 +  tr(H)/ n)/( 1 - (tr(H) + 2) / n)
  return(crit)
end

# Example:
#=
include("auxiliary function.jl")
 n = 1000
 mu = [0.0, 0]
 sigma = [1.0, 0.5, 0.5, 1.0]
 sigma = reshape(sigma, 2, 2)
 data = dgp(n, 1, mu, sigma)
 AIC_h = hurvic_AIC(data.y, data.x, bw = 0.1)
 AIC_iv_h = iv_AIC(data.y, data.x, data.z; bw = 0.1, type = "ll")
=#

# Function that computes the optimal bandwidth by minimizing to the iv_AIC function. Still in the works, use at your own risk.
function min_AIC_iv( expl, endo, instru; lower = 0.01, upper = 50.0, type = "lc") #, loo = FALSE )
   # bboptimize(iv_AIC; SearchRange = [lower, upper], NumDimensions = 1, expl = expl, endo = endo, instru = instru, type = type)   # Pkg.add("BlackBoxOptim")  DE type algorithm
   n = size(expl, 1)
   # H = kernel_hat_matrix(endo, instru, bw = h, type = type) #, loo)
   # # In the IV case, the effective hat matrix is (X'H'X)^(-1) X'H'
   # H = endo*inv(endo'*H'*endo)*endo'*H'
   # sig2 = ( expl'*( Matrix{Float64}(I, n, n) - H )'*( Matrix{Float64}(I, n, n) - H )*expl ) / n
   # # crit = log(sig2) + (1 +  tr(H)/ n)/( 1 - (tr(H) + 2) / n)

   function obj(h)  # I define the criterion function directly inside the min_AIC_iv function, so that it depends on the bandwidth only. Makes the minimization code easier
       # n = size(expl, 1)
        H = kernel_hat_matrix(endo, instru, bw = h, type = type) #, loo)
       # # In the IV case, the effective hat matrix is (X'H'X)^(-1) X'H'
        H = endo*inv(endo'*H'*endo)*endo'*H'
        sig2 = ( expl'*( Matrix{Float64}(I, n, n) - H )'*( Matrix{Float64}(I, n, n) - H )*expl ) / n
       crit = log(sig2) + (1 +  tr(H)/ n)/( 1 - (tr(H) + 2) / n)
       return crit
   end
   # With a classical optimizer
   sol = optimize(obj, lower, upper, Brent() )
   minimum = Optim.minimum(sol)
   minimizer = Optim.minimizer(sol)
   # With a Differential Evolution algorithm
   # sol = bboptimize(obj; SearchRange = (0.01, 50.0), NumDimensions = 1  )
   # minimum = best_fitness(sol)
   # minimizer = best_candidate(sol)
  # convergence = sol$convergence
  return(opt_obj = minimum, opt_h = minimizer)
end

# Computes the optimal bandwidth and estiamtes the linear iv model with it. Still in the works, use at your own risk.
function TV_kernel_iv(expl, endo, instru; option = "lc")
    if option != "lc" && option != "ll"
    error("option should be lc or ll ! Careful mate !")
    end
   sol = min_AIC_iv(expl, endo, instru, lower = 0.0001, upper = 100.0, type = option)
   minimum = sol.opt_obj
   opt_h = sol.opt_h
   bt_hat = kernel_tv( endo, expl, instru, option, opt_h)
   return (bt_hat = bt_hat, opt_h = opt_h, opt_obj = minimum)
end

# Example:
# mu = c(0,0)
# VCOV = matrix(c( 1 , 0.5 ,
#                  0.5 , 1 ), nrow = 2, ncol = 2)
# errors = MASS::mvrnorm(100, mu, Sigma = VCOV )
# z = rnorm(100)
# x = z + errors[, 2]
# y = 2*x + errors[, 1]
#
# iv_AIC(expl = y, endo = x, instru = z, h = 0.1)
# iv_AIC(expl = y, endo = x, instru = z, h = 0.1, loo = TRUE)
# iv_AIC(expl = y, endo = x, instru = z, h = 0.1, type = "ll")
# iv_AIC(expl = y, endo = x, instru = z, h = 0.1, type = "ll", loo = TRUE)
