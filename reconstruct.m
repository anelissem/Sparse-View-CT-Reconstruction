function [x, info] = reconstruct(prob)

    t_start = tic;

    b = prob.b(:);
    N = prob.N;
    angles = prob.angles(:);
    n_det = prob.n_det;

    if isfield(prob, 'sigma')
        sigma = prob.sigma;
    else
        sigma = 0;
    end

    if isfield(prob, 'x_min')
        xmin = prob.x_min;
    else
        xmin = 0;
    end

    if isfield(prob, 'x_max')
        xmax = prob.x_max;
    else
        xmax = inf;
    end

    geom = build_geom(N, angles, n_det);
    nang = numel(angles);
    sparsity = nang / N;

    if N <= 64
        max_time = 30;
        outer_max = 28;
        inner_iters = 14;
        lambda_l2 = 6e-5;
        lambda_tv = max(5e-5, 0.18 * sigma);
    elseif N <= 128 && sparsity >= 0.4
        max_time = 85;
        outer_max = 54;
        inner_iters = 20;
        lambda_l2 = 2e-4;
        lambda_tv = max(4e-4, 0.45 * sigma);
    elseif N <= 128
        max_time = 95;
        outer_max = 58;
        inner_iters = 22;
        lambda_l2 = 8e-4;
        lambda_tv = max(1.5e-3, 0.65 * sigma);
    else
        max_time = 102;
        outer_max = 54;
        inner_iters = 20;
        lambda_l2 = 1.5e-3;
        lambda_tv = max(2.5e-3, 0.75 * sigma);
    end

    Atb = adjoint_cached(b, N, n_det, geom);
    AAtb = apply_ata_cached(Atb, N, n_det, geom, 0);
    alpha0 = dot(Atb, Atb) / max(dot(Atb, AAtb), 1e-14);
    x = project_box(alpha0 * Atb, xmin, xmax);

    r = Atb - apply_ata_cached(x, N, n_det, geom, lambda_l2);
    p = r;
    rsold = dot(r, r);
    warm_iters = min(10, inner_iters);

    for i = 1:warm_iters
        Ap = apply_ata_cached(p, N, n_det, geom, lambda_l2);
        alpha = rsold / max(dot(p, Ap), 1e-14);
        x = project_box(x + alpha * p, xmin, xmax);
        r = r - alpha * Ap;
        rsnew = dot(r, r);
        if rsnew < 1e-18
            break;
        end
        p = r + (rsnew / max(rsold, 1e-14)) * p;
        rsold = rsnew;
    end

    total_iters = warm_iters;

    for outer = 1:outer_max
        if toc(t_start) > max_time
            break;
        end

        x = project_box(x, xmin, xmax);
        gamma = max(1e-5, 0.012 * mean(abs(x)) + 0.002 * max(abs(x)));
        W = compute_huber_weights(x, N, gamma);

        Ax = forward_cached(x, n_det, geom);
        grad_data = adjoint_cached(Ax - b, N, n_det, geom);
        grad_tv = apply_Lw(x, W, N);

        r = -(grad_data + lambda_tv * grad_tv + lambda_l2 * x);
        p = r;
        rsold = dot(r, r);

        for inner = 1:inner_iters
            if toc(t_start) > max_time
                break;
            end

            Ap = apply_ata_cached(p, N, n_det, geom, 0) + lambda_tv * apply_Lw(p, W, N) + lambda_l2 * p;
            pAp = dot(p, Ap);
            if pAp <= 1e-14 || ~isfinite(pAp)
                break;
            end

            alpha = rsold / pAp;
            x = project_box(x + alpha * p, xmin, xmax);
            r = r - alpha * Ap;
            rsnew = dot(r, r);
            total_iters = total_iters + 1;

            if sqrt(rsnew) < 1e-7 * max(1, norm(x))
                break;
            end

            beta = rsnew / max(rsold, 1e-14);
            p = r + beta * p;
            rsold = rsnew;
        end

        lambda_tv = lambda_tv * 0.96;
    end

    x = project_box(x, xmin, xmax);

    Ax = forward_cached(x, n_det, geom);
    sc = dot(Ax, b) / max(dot(Ax, Ax), 1e-14);
    if isfinite(sc) && sc > 0
        x = project_box(sc * x, xmin, xmax);
    end

    img = reshape(x, N, N);
    img = edge_preserving_smooth(img, N, nang, sigma, xmin, xmax);
    x = project_box(img(:), xmin, xmax);

    info.iters = total_iters;
end

function geom = build_geom(N, angles, n_det)
    [X, Y] = meshgrid(linspace(-N/2 + 0.5, N/2 - 0.5, N), linspace(-N/2 + 0.5, N/2 - 0.5, N));
    X = X(:);
    Y = Y(:);
    s_min = -N * sqrt(2) / 2;
    s_max = N * sqrt(2) / 2;
    bin_width = (s_max - s_min) / n_det;
    nang = numel(angles);
    geom.pix = cell(nang, 1);
    geom.bin = cell(nang, 1);
    for a = 1:nang
        s = X * cos(angles(a)) + Y * sin(angles(a));
        idx = round((s - s_min) / bin_width) + 1;
        valid = find(idx >= 1 & idx <= n_det);
        geom.pix{a} = valid;
        geom.bin{a} = idx(valid);
    end
end

function b = forward_cached(x, n_det, geom)
    nang = numel(geom.pix);
    b = zeros(nang * n_det, 1);
    for a = 1:nang
        pix = geom.pix{a};
        bins = geom.bin{a};
        b((a-1)*n_det + (1:n_det)) = accumarray(bins, x(pix), [n_det, 1]);
    end
end

function y = adjoint_cached(b, N, n_det, geom)
    y = zeros(N * N, 1);
    nang = numel(geom.pix);
    for a = 1:nang
        pix = geom.pix{a};
        bins = geom.bin{a};
        ba = b((a-1)*n_det + (1:n_det));
        y(pix) = y(pix) + ba(bins);
    end
end

function y = apply_ata_cached(x, N, n_det, geom, lambda)
    Ax = forward_cached(x, n_det, geom);
    y = adjoint_cached(Ax, N, n_det, geom) + lambda * x;
end

function x = project_box(x, xmin, xmax)
    x = max(x, xmin);
    if isfinite(xmax)
        x = min(x, xmax);
    end
end

function W = compute_huber_weights(x, N, gamma)
    X = reshape(x, N, N);
    Dx = [diff(X, 1, 2), zeros(N, 1)];
    Dy = [diff(X, 1, 1); zeros(1, N)];
    W = 1 ./ sqrt(Dx.^2 + Dy.^2 + gamma^2);
end

function Lx = apply_Lw(x, W, N)
    X = reshape(x, N, N);
    Dx = [diff(X, 1, 2), zeros(N, 1)];
    Dy = [diff(X, 1, 1); zeros(1, N)];
    Wx = W .* Dx;
    Wy = W .* Dy;
    divX = zeros(N, N);
    divX(:, 1) = -Wx(:, 1);
    divX(:, 2:N-1) = Wx(:, 1:N-2) - Wx(:, 2:N-1);
    divX(:, N) = Wx(:, N-1);
    divY = zeros(N, N);
    divY(1, :) = -Wy(1, :);
    divY(2:N-1, :) = Wy(1:N-2, :) - Wy(2:N-1, :);
    divY(N, :) = Wy(N-1, :);
    Lx = divX(:) + divY(:);
end

function u = edge_preserving_smooth(u, N, nang, sigma, xmin, xmax)
    if N <= 64
        steps = 1;
    elseif N <= 128 && nang < 30
        steps = 2;
    else
        steps = 1;
    end

    scale = max(u(:)) - min(u(:));
    if ~isfinite(scale) || scale <= eps
        return;
    end

    kappa = max(0.035 * scale, 1e-6);
    tau = 0.045 + 0.015 * (sigma > 0);

    for t = 1:steps
        north = [u(1, :); u(1:end-1, :)] - u;
        south = [u(2:end, :); u(end, :)] - u;
        west = [u(:, 1), u(:, 1:end-1)] - u;
        east = [u(:, 2:end), u(:, end)] - u;
        cn = exp(-(north / kappa).^2);
        cs = exp(-(south / kappa).^2);
        cw = exp(-(west / kappa).^2);
        ce = exp(-(east / kappa).^2);
        u = u + tau * (cn .* north + cs .* south + cw .* west + ce .* east);
        u = reshape(project_box(u(:), xmin, xmax), N, N);
    end
end