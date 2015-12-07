using PyPlot

type WaveSolution2D

  u::Array{Float64, 3}
  t_n::Int
  c::Float64   # wave speed
  dx::Float64  # grid step
  dy::Float64  # grid step
  dt::Float64  # time step

  # temporary storage for derivative computations
  u_xx::Array{Float64, 2}
  u_yy::Array{Float64, 2}
  u_tt::Array{Float64, 2}

  # inner constructor
  function WaveSolution2D(n_x, n_y, dx, dy, dt, c)
    return new(zeros(Float64, (n_x, n_y, 3)), 0, c, dx, dy, dt,
      zeros(Float64, (n_x, n_y)), zeros(Float64, (n_x, n_y)),
      zeros(Float64, (n_x, n_y)))
  end
end

# one step of wave equation solution
# assumes that there's already at least two frames!
function wave_eqn_step!(soln::WaveSolution2D)

  if soln.t_n < 2
    error("Need two temporal bits of information!")
  end

  n_x, n_y, n_t = size(soln.u)

  # compute spatial x derivatives (central differences)
  for x = 1:n_x
    for y = 1:n_y

      if x >= 2 && x <= n_x - 1
        soln.u_xx[x, y] = (soln.u[x + 1, y, 2] -
          2 * soln.u[x, y, 2] + soln.u[x - 1, y, 2]) / (soln.dx^2)
      end

      if y >= 2 && y <= n_y - 1
        soln.u_yy[x, y] = (soln.u[x, y + 1, 2] -
          2 * soln.u[x, y, 2] + soln.u[x, y - 1, 2]) / (soln.dy^2)
      end
    end
  end

  # you know, you don't actually have to store the second spatial derivatives
  # you can do it all in place! regardless,
  u_tt = 0.
  for x = 1:n_x
    for y = 1:n_y

      u_tt = soln.c^2 * (soln.u_xx[x, y] + soln.u_yy[x, y])
      soln.u[x, y, 3] = soln.dt^2 * u_tt + 2 * soln.u[x, y, 2] -
        soln.u[x, y, 1]
    end
  end

  # we also need to apply the boundary conditions
  soln.u[:, 1, 3] = zeros(n_x)
  soln.u[:, n_y, 3] = soln.u[:, n_y - 1, 3]
  soln.u[1, :, 3] = soln.u[2, :, 3]
  soln.u[n_x, :, 3] = soln.u[n_x - 1, :, 3]

  # shift everything backwards
  soln.u[:, :, 1] = soln.u[:, :, 2]
  soln.u[:, :, 2] = soln.u[:, :, 3]

  soln.t_n += 1
end

function init_solution!(soln::WaveSolution2D, u_initial::Array{Float64, 2},
  u_t_initial::Array{Float64, 2})

  n_x, n_y, n_t = size(soln.u)
  m_x, m_y = size(u_initial)

  if m_x != n_x || m_y != n_y
    error("Initial distribution dimensions don't match with solution.")
  end

  soln.u[:, :, 1] = u_initial
  soln.u[:, :, 2] = u_initial + soln.dt * u_t_initial
  soln.u[:, :, 3] = soln.u[:, :, 2]
  soln.t_n = 2
end

# return a number proportional to the energy
function wave_energy(soln::WaveSolution2D)

  # get u_t
  u_t = (soln.u[:, :, 2] - soln.u[:, :, 1]) / soln.dt

  # sum the squares
  return sum(u_t .^ 2)
end

function main()

  n_x = 100 ; n_y = 100
  dx = 0.001 ; dy = 0.001
  dt = 0.0005 ; c = 0.05

  # our data
  soln = WaveSolution2D(n_x, n_y, dx, dy, dt, c)

  # initial distribution
  u_initial = zeros(Float64, (n_x, n_y))
  # u_initial[45:55, 2] = 1

  x = collect(linspace(0, n_x - 1, n_x))
  u_initial[:, 2] = exp(-0.001 .* (x - n_x / 2 + 20) .^ 2) + exp(-0.01 * (x - n_x / 2) .^ 2)

  # ax = linspace(0, (n_x - 1) * dx, n_x)
  # xx = [i for i in ax, j in ax]
  # yy = [j for i in ax, j in ax]
  # u_initial = 10 * exp(-100000000.*(xx - (n_x .* dx ./ 2)).^2 - 100000000.*(yy - (n_y .* dy ./ 2)).^2)

  u_t_initial = zeros(Float64, (n_x, n_y))

  init_solution!(soln, u_initial, u_t_initial)

  n = 5000
  energy = zeros(n)

  # plot the data
  for i = 1:n
    if i % 10 == 1
      cla()
      imshow(soln.u[:, :, 2], interpolation="none", cmap="gray", clim=(-0.08, 0.08))
      # plot(soln.u[:, 2, 2])
      # ylim(-1, 1)
      println(maximum(soln.u[:, :, 2]))
      println(minimum(soln.u[:, :, 2]))
      println("Wave energy: ", string(wave_energy(soln)))
      # pause(1)
    end

    energy[i] = wave_energy(soln)

    wave_eqn_step!(soln)
  end

  fig = figure()
  plot(energy)
end
