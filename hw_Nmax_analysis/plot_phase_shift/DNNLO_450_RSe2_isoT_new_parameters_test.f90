! --- BEGIN AUTO-GENERATED ---
!   CMD: ./../nucleon-scattering/make_chp_init_code ./interactions/DNNLO_450-RSe2_isoT.ini +potential_name=DNNLO_450_RSe2_isoT
!   nsopt version: v1.8-112-g5c755c2-dirty

! SUITABLE V3 parameters:
!   c_D: 7.90000000000000036D-01
!   c_E: 1.70000000000000012D-02

subroutine chp_preset_DNNLO_450_RSe2_isoT_new
    use idaho_chiral_potential

    implicit none

    call initialize_chiral_potential

    ! GENERAL PARAMETERS AND CONSTANTS
    ! (proton, nucleon, neutron)
    call chp_set_mass_nucleon((/9.38272046000000046D+02, 9.38918267117927371D+02, 9.39565379000000007D+02/))
    ! (pi-, pi, pi+)
    call chp_set_mass_pion((/1.39570179999999993D+02, 1.34976599999999991D+02, 1.39570179999999993D+02/))

    ! delta mass in MeV 
    call chp_set_mass_delta(1.23200000000000000D+03)

    call chp_set_chiral_order(NNLO)
    call chp_set_chiral_mode(chiral_mode_EM2015)
    call chp_set_reg("SF", 7.00000000000000000D+02)

    ! leading pion-nucleon-delta coupling 
    call chp_set_hA(1.39999999999999991D+00)

    call chp_set_gA(1.28899999999999992D+00)
    call chp_set_fpi(9.22000000000000028D+01)
    call chp_set_Lambda(4.50000000000000000D+02)
    call chp_set_contact_format("PW")
    ! SET INCLUDED DIAGRAMS / TERMS

    ! LO contacts
    call chp_set_chiral_Ct_CIB(1) ! Use CIB contacts
    call chp_set_CIB_LO_contact(1, -1, -0.33871610D+00 ) ! Ct_1S0pp
    call chp_set_CIB_LO_contact(2, -1, -0.28605460D+00 ) ! Ct_3S1pp
    call chp_set_CIB_LO_contact(1,  0, -0.33982408D+00 ) ! Ct_1S0np
    call chp_set_CIB_LO_contact(2,  0, -0.28605460D+00 ) ! Ct_3S1np
    call chp_set_CIB_LO_contact(1,  1, -0.33847639D+00 ) ! Ct_1S0nn
    call chp_set_CIB_LO_contact(2,  1, -0.28605460D+00 ) ! Ct_3S1nn

    ! NLO contacts
    call chp_set_chiral_C(1) ! Use
    call chp_set_NLO_contact(1, 2.52024031D+00 ) ! C_1S0
    call chp_set_NLO_contact(2, 0.64477023D+00 ) ! C_3P0
    call chp_set_NLO_contact(3, 0.18364315D+00 ) ! C_1P1
    call chp_set_NLO_contact(4, -0.90129773D+00 ) ! C_3P1
    call chp_set_NLO_contact(5, 1.31945937D+00 ) ! C_3S1
    call chp_set_NLO_contact(6, 0.72331122D+00 ) ! C_3S1-3D1
    call chp_set_NLO_contact(7, -0.88829717D+00 ) ! C_3P2

    ! N3LO contacts
    call chp_set_chiral_D(0) ! Do not use

    ! Set needed ci and di, if any
    call chp_set_c1(-7.39999999999999991D-01)
    call chp_set_c3(-6.50000000000000022D-01)
    call chp_set_c4(9.59999999999999964D-01)

    ! sub-leading delta excitation require c2
    call chp_set_c2(-4.89999999999999991D-01)

    ! sub-leading piND (b3+b8) can be absorbed into leading  and sub-leading piN LECs 
    call chp_set_b3(0.00000000000000000D+00)
    call chp_set_b8(0.00000000000000000D+00)

    ! Parameters for the NNN force
    !call chp_set_c_D(7.90000000000000036D-01)
    !call chp_set_c_E(1.70000000000000012D-02)

    ! Set regulator parameter n
    call chp_set_1PE_reg_par(3.0D0)
    call chp_set_2PE_reg_par(3.0D0)
    call chp_set_LO_contact_reg_par(1, 3.0D0) ! Ct_1S0
    call chp_set_LO_contact_reg_par(2, 3.0D0) ! Ct_3S1
    call chp_set_NLO_contact_reg_par(1, 3.0D0) ! C_1S0
    call chp_set_NLO_contact_reg_par(2, 3.0D0) ! C_3P0
    call chp_set_NLO_contact_reg_par(3, 3.0D0) ! C_1P1
    call chp_set_NLO_contact_reg_par(4, 3.0D0) ! C_3P1
    call chp_set_NLO_contact_reg_par(5, 3.0D0) ! C_3S1
    call chp_set_NLO_contact_reg_par(6, 3.0D0) ! C_3S1-3D1
    call chp_set_NLO_contact_reg_par(7, 3.0D0) ! C_3P2
    call chp_set_N3LO_contact_reg_par(1, 3.0D0) ! Dh_1S0
    call chp_set_N3LO_contact_reg_par(2, 3.0D0) ! D_1S0
    call chp_set_N3LO_contact_reg_par(3, 3.0D0) ! D_3P0
    call chp_set_N3LO_contact_reg_par(4, 3.0D0) ! D_1P1
    call chp_set_N3LO_contact_reg_par(5, 3.0D0) ! D_3P1
    call chp_set_N3LO_contact_reg_par(6, 3.0D0) ! Dh_3S1
    call chp_set_N3LO_contact_reg_par(7, 3.0D0) ! D_3S1
    call chp_set_N3LO_contact_reg_par(8, 3.0D0) ! D_3D1
    call chp_set_N3LO_contact_reg_par(9, 3.0D0) ! Dh_3S1-3D1
    call chp_set_N3LO_contact_reg_par(10, 3.0D0) ! D_3S1-3D1
    call chp_set_N3LO_contact_reg_par(11, 3.0D0) ! D_1D2
    call chp_set_N3LO_contact_reg_par(12, 3.0D0) ! D_3D2
    call chp_set_N3LO_contact_reg_par(13, 3.0D0) ! D_3P2
    call chp_set_N3LO_contact_reg_par(14, 3.0D0) ! D_3P2-3F2
    call chp_set_N3LO_contact_reg_par(15, 3.0D0) ! D_3D3

    ! Set pion exchange contributions
    ! Basic 1PE
    call chp_set_chiral_1PE(1) ! Use
    ! CIB effects in 1PE
    call chp_set_chiral_1PE_CIB(1) ! Use
    ! pion-gamma exchange
    call chp_set_chiral_1PE_gamma(0) ! Do not use
    ! Relativistic corrections to 1PE
    call chp_set_chiral_1PE_relcorr(0) ! Do not use

    ! Leading 2PE
    call chp_set_chiral_2PE_1loop_0(1) ! Use

    ! 1-loop 2PE proportional to ci
    call chp_set_chiral_2PE_1loop_d(1) ! Use

    ! 1-loop 2PE proportional to 1/M_N (relativistic corrections)
    call chp_set_chiral_2PE_1loop_r(0) ! Do not use

    ! 1-loop 2PE proportional to ci*cj)
    call chp_set_chiral_2PE_1loop_dd(0) ! Do not use

    ! 1-loop 2PE proportional to ci/M_N)
    call chp_set_chiral_2PE_1loop_dr(0) ! Do not use

    ! 1-loop 2PE proportional to 1/M_N^2)
    call chp_set_chiral_2PE_1loop_rr(0) ! Do not use

    ! 2-loop 2PE
    call chp_set_chiral_2PE_2loop(0) ! Do not use
    ! Contributions to 2-loop 2PE that do not have analytical expressions
    call chp_set_chiral_2PE_2loop_int(0) ! Do not use

    ! Use correct nucleon mass in 2PE relativistic corrections
    call chp_set_chiral_2PE_CSB_correct_mass(0) ! Do not use

    ! Use minimal relativity
    call chp_set_chiral_minimal_relativity(1) ! Use

    ! Use Kamada-Glockle transform
    call chp_set_chiral_kamada_glockle_transform(0) ! Do not use

    ! Include explicit delta excitations
    call chp_set_chiral_delta_2PE(1)
    ! Exclude certain explicit delta excitation diagrams by uncommenting  (included by default)
    !call chp_set_chiral_delta_2PE_leading_triangle(0)      
    !call chp_set_chiral_delta_2PE_leading_single_box(0)    
    !call chp_set_chiral_delta_2PE_leading_double_box(0)    
    !call chp_set_chiral_delta_2PE_subleading_triangle(0)   
    !call chp_set_chiral_delta_2PE_subleading_single_box(0) 
    !call chp_set_chiral_delta_2PE_subleading_double_box(0) 

    call chp_set_units_and_derive_constants

end subroutine
! --- END AUTO-GENERATED ---
