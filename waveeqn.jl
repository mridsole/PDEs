using PyPlot

type WaveSolution2D

  u::Array{Float64, 3}
  t_n::Int
  c::Float64   # wave speed
  dx::Float64  # grid step
  dy::Float64  # grid step
  dt::Float64  # time step
  BUFFER_SIZE::Int

  # temporary storage for derivative computations
  u_xx::Array{Float64, 2}
  u_yy::Array{Float64, 2}
  u_tt::Array{Float64, 2}

  # inner constructor
  function WaveSolution2D(n_x, n_y, dx, dy, dt, c)
    return new(zeros(Float64, (n_x, n_y, 0)), 0, c, dx, dy, dt, 512,
      zeros(Float64, (n_x, n_y)), zeros(Float64, (n_x, n_y)),
      zeros(Float64, (n_x, n_y)))
  end
end

# one step of wave equation solution
# assumes that there's already at least two frames!
function wave_eqn_step!(soln::WaveSolution2D)

  # allocate some more memory if necessary
  n_x, n_y, n_t = size(soln.u)
  if n_t == soln.t_n
    soln.u = cat(3, soln.u, zeros(Float64, (n_x, n_y, soln.BUFFER_SIZE)))
  end

  if soln.t_n < 2
    error("Need two temporal bits of information!")
  end

  # compute spatial x derivatives (central differences)
  for x = 1:n_x
    for y = 1:n_y

      if x >= 2 && x <= n_x - 1
        soln.u_xx[x, y] = (soln.u[x + 1, y, soln.t_n] -
          2 * soln.u[x, y, soln.t_n] + soln.u[x - 1, y, soln.t_n]) / (soln.dx^2)
      end

      if y >= 2 && y <= n_y - 1
        soln.u_yy[x, y] = (soln.u[x, y + 1, soln.t_n] -
          2 * soln.u[x, y, soln.t_n] + soln.u[x, y - 1, soln.t_n]) / (soln.dy^2)
      end
    end
  end

  # you know, you don't actually have to store the second spatial derivatives
  # you can do it all in place! regardless,
  u_tt = 0.
  for x = 1:n_x
    for y = 1:n_y

      u_tt = soln.c^2 * (soln.u_xx[x, y] + soln.u_yy[x, y])
      soln.u[x, y, soln.t_n + 1] = soln.dt^2 * u_tt + 2 * soln.u[x, y, soln.t_n] -
        soln.u[x, y, soln.t_n - 1]
    end
  end

  # we also need to apply the boundary conditions
  soln.u[:, 1, soln.t_n + 1] = zeros(n_x)
  soln.u[:, n_y, soln.t_n + 1] = zeros(n_x)
  soln.u[1, :, soln.t_n + 1] = zeros(n_y)
  soln.u[n_x, :, soln.t_n + 1] = zeros(n_y)

  soln.t_n += 1
end

function init_solution!(soln::WaveSolution2D, u_initial::Array{Float64, 2},
  u_t_initial::Array{Float64, 2})

  n_x, n_y, n_t = size(soln.u)
  m_x, m_y = size(u_initial)

  if m_x != n_x || m_y != n_y
    error("Initial distribution dimensions don't match with solution.")
  end

  # allocate some more memory if necessary
  if n_t <= 1
    soln.u = cat(3, soln.u, zeros(Float64, (n_x, n_y, soln.BUFFER_SIZE)))
  end

  soln.u[:, :, 1] = u_initial
  soln.u[:, :, 2] = u_initial + soln.dt * u_t_initial
  soln.t_n = 2
end

# return a number proportional to the energy
function wave_energy(soln::WaveSolution2D)

  # get u_t
  u_t = (soln.u[:, :, soln.t_n] - soln.u[:, :, soln.t_n - 1]) / soln.dt

  # sum the squares
  return sum(u_t .^ 2)
end

function main()

  n_x = 100 ; n_y = 100
  dx = 0.0001 ; dy = 0.0001
  dt = 0.00001 ; c = 1

  # our data
  soln = WaveSolution2D(n_x, n_y, dx, dy, dt, c)

  # initial distribution
  # u_initial = zeros(Float64, (n_x, n_y))
  # u_initial[41:60, 41:60] = 0.01 * ones(Float64, (20, 20))
  # u_initial[2, 2] = 1

  ax = linspace(0, (n_x - 1) * dx, n_x)
  xx = [i for i in ax, j in ax]
  yy = [j for i in ax, j in ax]
  u_initial = 10 * exp(-100000000.*(xx - (n_x .* dx ./ 2)).^2 - 100000000.*(yy - (n_y .* dy ./ 2)).^2)

  u_t_initial = zeros(Float64, (n_x, n_y))

  init_solution!(soln, u_initial, u_t_initial)

  # plot the data
  for i = 1:100000
    if i % 10 == 1
      cla()
      imshow(soln.u[:, :, i], interpolation="none", cmap="gray", clim=(-0.01, 0.01))
      println(maximum(soln.u[:, :, i]))
      println(minimum(soln.u[:, :, i]))
      println("Wave energy: ", string(wave_energy(soln)))
      pause(1)
    end

    wave_eqn_step!(soln)
  end
end
