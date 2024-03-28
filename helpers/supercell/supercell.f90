module supercell
  use linalg, only: determinant_3x3_real, frobnorm_3x3_real, matinv3x3

  contains

    !> compute fractional positions
    pure function fractional(pos, inv_lattice) result(frac_pos)
      real*8, intent(in) :: pos(3,1), inv_lattice(3,3)
      real*8             :: frac_pos(3,1)

      frac_pos = matmul(inv_lattice, pos)

    end function

    !> Compute deviation from the target metric, given that cell is normed
    pure function get_deviation(cell, target_metric, norm) result(deviation)
      real*8, dimension(3,3), intent(in) :: cell
      real*8, dimension(3,3), intent(in) :: target_metric
      logical,      optional, intent(in) :: norm
      logical                            :: nrm
      real*8                             :: ncell(3,3), deviation

      nrm = .true.
      if (present(norm)) nrm = norm

      ! Normalize the input lattice
      if (nrm) then
        ncell = cell * (determinant_3x3_real(cell) / determinant_3x3_real(target_metric))**(-1./3.)
      else
        ncell = cell
      end if

      deviation = frobnorm_3x3_real(ncell - target_metric)

    end function

    !> Find optimal supercell matrix to realize supercell lattice of given
    !> metric, e.g., cubic.
    function find_optimal_cell(cell, target_metric, target_size, deviation, &
        lower_limit, upper_limit, verbose) result(smatrix)
      real*8,  intent(in)   :: cell(3,3)
      real*8,  intent(in)   :: target_metric(3,3)
      real*8,  intent(in)   :: target_size, deviation
      integer, intent(in)   :: lower_limit, upper_limit
      logical, intent(in)   :: verbose
      integer               :: smatrix(3,3)
      optional              :: lower_limit, upper_limit, deviation, verbose
      integer               :: llim, ulim
      logical               :: vrbs, found
      integer               :: i1, i2, i3, i4, i5, i6, i7, i8, i9, nn
      integer               :: initial_P(3,3), P(3,3), dP(3,3)
      real*8                :: dev, norm, score, best_score, ideal_P(3,3), ccell(3,3)

      ! options and defaults
      dev  = 0.1d0
      llim = -2
      ulim =  2
      vrbs = .false.
      if(present(lower_limit)) llim = lower_limit
      if(present(upper_limit)) ulim = upper_limit
      if(present(deviation))   dev  = deviation
      if(present(verbose))     vrbs = verbose

      ! initialize matrices and values
      smatrix    = -1
      score      = 1.e7
      best_score = 1.e6
      ccell      = cell
      found      = .false.
      P          = 0

      if (vrbs) then
        write(*,*) 'Settings:'
        write (*,"(A,/,3(F7.2))") 'Input cell:   ', cell
        write (*,"(A,/,3(F7.2))") 'Target metric:', target_metric
        write (*,"(A,(F7.2))")    'Target size:  ', target_size
        write (*,"(A,(F7.2))")    'Allowed dev.: ', dev
        write (*,"(A,2(I3))")     'Limits:       ', llim, ulim
      end if

      ! Normalize the input lattice
      norm = (target_size * determinant_3x3_real(cell) / determinant_3x3_real(target_metric))**(-1./3.)
      ccell = norm * cell

      if (vrbs) write(*,"(A,(F7.3))") 'Normalization factor: ', norm
      if (vrbs) write (*,"(A,/,3(F7.2))") 'Normed cell:  ', ccell

      ! Approximate the perfect smatrix
      ideal_P   = matmul(target_metric, matinv3x3(ccell))
      initial_P = nint(ideal_P)
      if (vrbs) write (*,"(A,/,3(F7.3))") 'Ideal P:   ' , ideal_P
      if (vrbs) write (*,"(A,/,3(I7))") 'Initial P: ' , initial_P

      !> Expand brute force
      do i1=llim, ulim
      do i2=llim, ulim
      do i3=llim, ulim
      do i4=llim, ulim
      do i5=llim, ulim
      do i6=llim, ulim
      do i7=llim, ulim
      do i8=llim, ulim
      do i9=llim, ulim
        dP(1,1:3) =  (/ i1, i2, i3 /)
        dP(2,1:3) =  (/ i4, i5, i6 /)
        dP(3,1:3) =  (/ i7, i8, i9 /)
        P = initial_P + dP
        nn = nint(determinant_3x3_real(real(P, 8)))
        ! Allow some deviation from target size (only increase)
        if ((nn < target_size  ) .or. (nn > (1.d0 + dev) * target_size)) cycle
        score = get_deviation(matmul(P, ccell), target_metric)
        ! Save the result if it was a good one
        if (score < best_score) then
          found = .true.
          best_score = score
          smatrix = P
        end if
      end do
      end do
      end do
      end do
      end do
      end do
      end do
      end do
      end do

      if (vrbs) then
        write (*,"(A,/,(F7.3))") 'Best score: ' , best_score
        write (*,"(A,/,3(I7))")  'P:          ' , smatrix
        write (*,"(A,/,F7.3,I5)")  'N_target, N:' , target_size, nint(determinant_3x3_real(real(smatrix, 8)))
      end if

      if (.not. found) write(*,*) 'No supercell matrix found.'

    end function

    !> Find lattice points by brute force enumeration
    function find_lattice_points(lattice, inv_superlattice, n_lattice_points, &
                                 max_iterations, tolerance) result(all_lattice_points)
      real*8,  intent(in)   :: lattice(3,3), inv_superlattice(3,3), tolerance
      integer, intent(in)   :: max_iterations, n_lattice_points
      real*8                :: lattice_points(3,n_lattice_points), lp(3,1), frac_lp(3,1)
      real*8                :: lattice_points_extended(3, 8*n_lattice_points)
      real*8                :: all_lattice_points(3, 9*n_lattice_points)
      integer               :: n1, n2, n3, counter, counter_ext


      ! ! initialize matrices and values
      lattice_points          = -10000.0d0
      lattice_points_extended = -10000.0d0
      lp                      = 0.0d0
      elp                     = 0.0d0

      ! if (vrbs) then
      !   write(*,*) 'Settings:'
      !   write (*,"(A,/,(I3))") 'n_lattice_points', n_lattice_points
      ! end if

      !> Expand brute force
      counter     = 0
      counter_ext = 0
      do n1 = -max_iterations, max_iterations
      do n2 = -max_iterations, max_iterations
      do n3 = -max_iterations, max_iterations

        lp =  dble(n1) * lattice(:, 1:1) &
            + dble(n2) * lattice(:, 2:2) &
            + dble(n3) * lattice(:, 3:3)

        frac_lp = fractional(lp, inv_superlattice)

        ! check if frac_lp is within supercell [0, 1)
        if     ((frac_lp(1, 1) > -tolerance) &
          .and. (frac_lp(2, 1) > -tolerance) &
          .and. (frac_lp(3, 1) > -tolerance) &
          .and. (frac_lp(1, 1) < 1 - tolerance) &
          .and. (frac_lp(2, 1) < 1 - tolerance) &
          .and. (frac_lp(3, 1) < 1 - tolerance)) then

            counter = counter + 1
            lattice_points(1:3, counter:counter) = lp(1:3, 1:1)

        ! check if frac_lp is within extended supercell [0, 1]
        else if     ((frac_lp(1, 1) > -tolerance) &
          .and. (frac_lp(2, 1) > -tolerance) &
          .and. (frac_lp(3, 1) > -tolerance) &
          .and. (frac_lp(1, 1) < 1 + tolerance) &
          .and. (frac_lp(2, 1) < 1 + tolerance) &
          .and. (frac_lp(3, 1) < 1 + tolerance)) then

            counter_ext = counter_ext + 1
            ! write (*,*) 'counter: ', counter
            ! write (*,*) 'counter_ext: ', counter_ext
            lattice_points_extended(1:3, counter_ext:counter_ext) = lp(1:3, 1:1)

        end if

        if (counter > n_lattice_points) then
          write(*,*) "Counter:          ", counter
          write(*,*) "N lattice_points: ", n_lattice_points
          exit
        end if

      end do
      end do
      end do

      all_lattice_points(:, 1:n_lattice_points) = lattice_points
      all_lattice_points(:, n_lattice_points + 1 : 9*n_lattice_points + 1) = &
        lattice_points_extended

    end function

    function remap_force_constants(positions, pairs, fc_in, map2prim, inv_lattice, &
      tol, eps) result(fc_out)
      real*8,  intent(in)   :: positions(:,:), pairs(:,:,:), fc_in(:,:,:,:)
      real*8,  intent(in)   :: inv_lattice(3, 3), tol, eps
      integer, intent(in)   :: map2prim(:)
      real*8                :: fc_out(size(positions, 1), size(positions, 1), 3, 3)
      real*8                :: r_12(3), r_12_frac(3)
      real*8                :: r_diff(3), ref_pair_frac(3), r_1(3), r_3(3)
      integer               :: natoms, uc_index, i1, i2, i3

      natoms = size(positions, 1)
      natoms_orig = size(pairs, 2)

      frac_lp = 0.0

      ! initialize matrices and values
      fc_out = 0.0d0

      do i1 = 1, natoms
        ! position of atoms i1
        r_1 = positions(i1, :)
        ! index of atom in primitive cell of which i1 is an image
        uc_index = map2prim(i1) + 1
        do i2 = 1, natoms_orig
          ! pair vector
          r_12 = r_1 + pairs(uc_index, i2, :)
          ! pair vector in fractional coord. w.r.t supercell
          r_12_frac = modulo(matmul(inv_lattice, r_12), 1.0d0)
          do i3 = 1, natoms
            ! reference pair vector in fractional coord. w.r.t supercell
            r_3 = positions(i3, :)
            ref_pair_frac = modulo( &
              modulo(matmul(inv_lattice, r_3), 1.0d0), 1.0d0 &
            )
            r_diff = abs(r_12_frac - ref_pair_frac)
            r_diff = r_diff - floor(r_diff + eps)
            if (sum(abs(r_diff)) < tol) then
              fc_out(i1, i3, :, :) = fc_out(i1, i3, :, :) + fc_in(uc_index, i2, :, :)
            end if
          end do
        end do
       end do

    end function

end module
