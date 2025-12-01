using LinearAlgebra

function csminwel(fcn, x0, H0, grad, crit, nit; kwargs...) 
    #= 
    This function accepts inputs the following inputs:
    fcn: Julia function describing the function to be minimized
    x0: Initial point
    H0: Initial Hessian
    grad: Gradient function (if available, otherwise pass "nothing")
    crit: Objective improvement threshold condition
    nit: Max. number of iterations before terminating
    =#
    
    
    nx = length(x0)
    x = copy(x0)
    H = Matrix(H0)
    NumGrad = isnothing(grad)
    f = fcn(x; kwargs...)

    if f > 1e50
        error("Bad initial parameter.")
    end

    g, badg = NumGrad ? numerical_gradient(fcn, x; kwargs...) : grad(x; kwargs...)

    itct = 0
    fcount = 0
    done = false
    
    # --- FIX START: Initialize retcode1 here ---
    retcode1 = 0 
    # --- FIX END ---

    while !done
        println("-----------------")
        println("Iteration: $itct")
        println("Function value: $f")
        println("x: $x")

        itct += 1

        # Because retcode1 was initialized above, this update affects the outer variable
        f1, x1, fc, retcode1 = csminit(fcn, x, f, g, badg, H; kwargs...)
        fcount += fc
        print("Return Code: $retcode1\nEval Count: $fcount\n")
        
        g1, badg1 = NumGrad ? numerical_gradient(fcn, x1; kwargs...) : grad(x1; kwargs...)

        if retcode1 != 1
            if retcode1 in [2, 4]
                wall1 = true
            else
                wall1 = badg1
            end

            if wall1 && nx > 1
                println("Cliff detected. Perturbing search direction.")
                H_cliff = H + Diagonal(rand(nx) .* diag(H))
                f2, x2, fc, retcode2 = csminit(fcn, x, f, g, badg, H_cliff; kwargs...)
                fcount += fc

                if f2 < f
                    g2, badg2 = NumGrad ? numerical_gradient(fcn, x2; kwargs...) : grad(x2; kwargs...)
                    wall2 = badg2
                    if wall2
                        println("Cliff again. Traversing.")
                        if norm(x2 - x1) < 1e-13
                            f3, x3, badg3, retcode3 = f, x, true, 101
                        else
                            g_cliff = ((f2 - f1) / (norm(x2 - x1)^2)) * (x2 - x1)
                            f3, x3, fc, retcode3 = csminit(fcn, x, f, g_cliff, false, Matrix(I(nx)); kwargs...)
                            fcount += fc
                            g3, badg3 = NumGrad ? numerical_gradient(fcn, x3; kwargs...) : grad(x3; kwargs...)
                        end
                    else
                         f3, x3, badg3, retcode3 = f, x, true, 101
                    end
                else
                    f3, x3, badg3, retcode3 = f, x, true, 101
                end
            else
                 f2, f3, badg2, badg3, retcode2, retcode3 = f, f, true, true, 101, 101
            end
        else
            f2, f3, f1, retcode2, retcode3 = f, f, f, retcode1, retcode1
        end

        if !badg && !badg1 && abs(f1 - f) >= crit
            H = bfgsi(H, g1 - g, x1 - x)
        end

        println("Improvement: $(f - f1)")

        if itct >= nit || abs(f - f1) < crit
            done = true
        end
        
        # Update state for next iteration
        f, x, g = f1, x1, g1
    end

    return f, x, g, H, itct, fcount, retcode1
end

# ------------------------------------ HELPER FUNCTIONS --------------------------------------
function numerical_gradient(fcn, x; kwargs...)
    eps = 1e-8
    g = zeros(length(x))
    for i in 1:length(x)
        x_forward = copy(x)
        x_backward = copy(x)
        x_forward[i] += eps
        x_backward[i] -= eps
        g[i] = (fcn(x_forward; kwargs...) - fcn(x_backward; kwargs...)) / (2 * eps)
    end
    return g, false
end

function csminit(fcn, x0, f0, g0, badg, H0; kwargs...)
    ANGLE = 0.005
    THETA = 0.3
    FCHANGE = 1000
    MINLAMB = 1e-9
    MINDFAC = 0.01
    fcount = 0
    lambda = 1.0
    xhat = x0
    fhat = f0
    g = g0
    gnorm = norm(g)

    if gnorm < 1e-12 && !badg
        return fhat, xhat, fcount, 1
    end

    dx = -H0 * g
    dxnorm = norm(dx)
    if dxnorm > 1e12
        dx *= FCHANGE / dxnorm
    end

    dfhat = dot(dx, g0)
    if !badg
        a = -dfhat / (gnorm * dxnorm)
        if a < ANGLE
            dx -= (ANGLE * dxnorm / gnorm + dfhat / (gnorm^2)) * g
            dx *= dxnorm / norm(dx)
            dfhat = dot(dx, g)
        end
    end

    done = false
    factor = 3.0
    shrink = true
    lambda_min = 0.0
    lambda_max = Inf
    lambda_peak = 0.0
    f_peak = f0
    lambdahat = 0.0

    while !done
        dxtest = x0 + lambda * dx
        f = fcn(dxtest; kwargs...)
        fcount += 1
        if f < fhat
            fhat = f
            xhat = dxtest
            lambdahat = lambda
        end
        shrink_signal = (!badg && (f0 - f < max(-THETA * dfhat * lambda, 0))) || (badg && (f0 - f) < 0)
        grow_signal = !badg && (lambda > 0 && (f0 - f > -(1 - THETA) * dfhat * lambda))

        if shrink_signal && (lambda > lambda_peak || lambda < 0)
            if lambda > 0 && (!shrink || lambda / factor <= lambda_peak)
                shrink = true
                factor = factor^0.6
                while lambda / factor <= lambda_peak
                    factor = factor^0.6
                end
                if abs(factor - 1) < MINDFAC
                    return fhat, xhat, fcount, if abs(lambda) < 4.0 2 else 7 end
                end
            end
            lambda = lambda / factor
            if abs(lambda) < MINLAMB
                if lambda > 0 && f0 <= fhat
                    lambda = -lambda * factor^6
                else
                    return fhat, xhat, fcount, if lambda < 0 6 else 3 end
                end
            end
        elseif grow_signal && lambda > 0 || (shrink_signal && (lambda <= lambda_peak && lambda > 0))
            if shrink
                shrink = false
                factor = factor^0.6
                if abs(factor - 1) < MINDFAC
                    return fhat, xhat, fcount, if abs(lambda) < 4.0 4 else 7 end
                end
            end
            if f < f_peak && lambda > 0
                f_peak = f
                lambda_peak = lambda
                if lambda_max <= lambda_peak
                    lambda_max = lambda_peak * factor^2
                end
            end
            lambda = lambda * factor
            if abs(lambda) > 1e20
                return fhat, xhat, fcount, 5
            end
        else
            done = true
            return fhat, xhat, fcount, if factor < 1.2 7 else 0 end
        end
    end
    return fhat, xhat, fcount, 0
end

function bfgsi(H, delta_g, delta_x) #-------------BFGS algorithm for hessian calculation------------
    rho = 1.0 / (dot(delta_g, delta_x))
    # Safety check for division by zero in BFGS update
    if isinf(rho) || isnan(rho)
        return H
    end
    V = I(length(delta_x)) - rho * (delta_x * delta_g')
    return V * H * V' + rho * (delta_x * delta_x')
end



function sims_optimize(fcn, x0, H0, grad, crit, nit; kwargs...) 
    println("STARTING OPTIMIZATION...")
    results=csminwel(fcn, x0, H0, grad, crit, nit)
    println("\nFINAL RESULTS:")
    println("Min Value: ", results[1])
    println("Min X: ", results[2])
    return results[1],results[2]
end