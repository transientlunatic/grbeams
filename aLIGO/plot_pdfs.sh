pickle1=angle_re_2022_delta_0p5.pickle
#pickle2=angle_re_2022_delta_1.pickle
pickle3=angle_re_2022_uniform.pickle
pickle4=angle_re_2022_jeffreys.pickle
../overlay_ADE_jets.py "${pickle1}|${pickle3}|${pickle4}" 2022 re
echo "done 22"

pickle1=angle_re_2016_delta_0p5.pickle
#pickle2=angle_re_2016_delta_1.pickle
pickle3=angle_re_2016_uniform.pickle
pickle4=angle_re_2016_jeffreys.pickle
../overlay_ADE_jets.py "${pickle1}|${pickle3}|${pickle4}" 2016 re
echo "done 16"

#   pickle1=angle_re_2022_delta,0.5_sim_theta-30.0_epsilon-0.5.pickle
#   pickle2=angle_re_2022_uniform_sim_theta-30.0_epsilon-0.5.pickle
#   pickle3=angle_re_2022_jeffreys_sim_theta-30.0_epsilon-0.5.pickle
#   ../overlay_ADE_jets.py "${pickle1}|${pickle2}|${pickle3}" 2022 30
#   echo "done 22"
#
#   pickle1=angle_re_2016_delta,0.5_sim_theta-30.0_epsilon-0.5.pickle
#   pickle2=angle_re_2016_uniform_sim_theta-30.0_epsilon-0.5.pickle
#   pickle3=angle_re_2016_jeffreys_sim_theta-30.0_epsilon-0.5.pickle
#   ../overlay_ADE_jets.py "${pickle1}|${pickle2}|${pickle3}" 2016 30
#   echo "done 16"



