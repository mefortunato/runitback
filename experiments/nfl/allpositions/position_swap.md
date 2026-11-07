### Position swap, ae
```
120 games; one change to the home team's depth chart vs its previous-game lineup (Δ home win, Δ margin in points, 95% CI):
  slot  change player                              Δ win         Δ margin
  QB    best   Josh Allen               +0.054 ± 0.014   +1.58 ± 0.37
  QB    worst  Davis Mills              -0.011 ± 0.013   -0.33 ± 0.39
  QB    new                             -0.154 ± 0.013   -4.91 ± 0.37
  QB    out                             -0.024 ± 0.016   -0.71 ± 0.45
  RB    best   Aaron Jones              +0.010 ± 0.011   +0.41 ± 0.26
  RB    worst  Ezekiel Elliott          -0.003 ± 0.011   -0.03 ± 0.26
  RB    new                             -0.024 ± 0.009   -0.74 ± 0.23
  RB    out                             -0.002 ± 0.011   -0.00 ± 0.27
  WR    best   Mike Evans               +0.015 ± 0.010   +0.51 ± 0.24
  WR    worst  Jerry Jeudy              -0.028 ± 0.008   -0.82 ± 0.22
  WR    new                             -0.016 ± 0.009   -0.49 ± 0.23
  WR    out                             -0.010 ± 0.010   -0.13 ± 0.25
  TE    best   Travis Kelce             +0.033 ± 0.009   +1.05 ± 0.24
  TE    worst  Drew Sample              -0.007 ± 0.010   -0.34 ± 0.22
  TE    new                             -0.027 ± 0.010   -0.83 ± 0.24
  TE    out                             -0.008 ± 0.011   -0.26 ± 0.24
  OL    best   Andrew Wylie             +0.038 ± 0.010   +1.30 ± 0.21
  OL    worst  Robert Jones             -0.030 ± 0.009   -0.90 ± 0.21
  OL    new                             -0.018 ± 0.008   -0.46 ± 0.19
  OL    out                             -0.008 ± 0.010   -0.13 ± 0.25
  DL    best   Nick Bosa                +0.016 ± 0.008   +0.49 ± 0.19
  DL    worst  Travon Walker            -0.020 ± 0.009   -0.38 ± 0.20
  DL    new                             -0.018 ± 0.008   -0.55 ± 0.17
  DL    out                             +0.002 ± 0.010   +0.07 ± 0.21
  LB    best   Myles Garrett            +0.008 ± 0.009   +0.28 ± 0.19
  LB    worst  Bobby Wagner             -0.016 ± 0.008   -0.41 ± 0.20
  LB    new                             -0.008 ± 0.009   -0.07 ± 0.21
  LB    out                             -0.001 ± 0.010   +0.04 ± 0.23
  DB    best   C.J. Gardner-Johnson     -0.004 ± 0.009   -0.09 ± 0.19
  DB    worst  Mike Jackson             -0.003 ± 0.008   -0.17 ± 0.17
  DB    new                             -0.005 ± 0.009   -0.16 ± 0.19
  DB    out                             -0.000 ± 0.009   -0.02 ± 0.19
  K     best   Brandon Aubrey           +0.005 ± 0.009   +0.15 ± 0.18
  K     worst  Jake Moody               +0.001 ± 0.008   +0.07 ± 0.21
  K     new                             -0.003 ± 0.009   -0.03 ± 0.19
  K     out                             +0.000 ± 0.009   +0.01 ± 0.19
  P     best   Corliss Waitman          -0.006 ± 0.008   -0.09 ± 0.19
  P     worst  Ethan Evans              -0.001 ± 0.008   -0.02 ± 0.18
  P     new                             -0.001 ± 0.008   -0.06 ± 0.18
  P     out                             -0.002 ± 0.009   -0.01 ± 0.20

4514 players with 16+ games: each one's own effect on a league-average play vs a replacement-level player (yards per play; defenders: yards taken away)
  QB  n= 122  sd Δyards 2.162   corr with direct value -0.00 (n=74)   best: Derek Carr, Drew Stanton, Patrick Mahomes   worst: Tony Romo, Landry Jones
  RB  n= 420  sd Δyards 2.368   corr with direct value -0.04 (n=166)   best: T.J. Logan, Duke Johnson, Elijah McGuire   worst: Michael Carter, Khalil Herbert
  WR  n= 549  sd Δyards 2.192   corr with direct value +0.05 (n=272)   best: De'Anthony Thomas, DJ Chark, Malik Heath   worst: Rashod Bateman, Kayshon Boutte
  TE  n= 319  sd Δyards 1.964   corr with direct value -0.01 (n=163)   best: Vernon Davis, Alex Smith, Mychal Rivera   worst: Jacob Hollister, Charlie Woerner
  OL  n= 749  sd Δyards 2.334   corr with direct value +0.02 (n=405)   best: John Urschel, Cody Mauch, Anthony Steen   worst: Alex Cappa, Brian De La Puente
  DL  n= 663  sd Δyards 1.989   corr with direct value +0.01 (n=367)   best: Montez Sweat, Derek Barnett, Shaq Lawson   worst: Greg Gaines, R.J. McIntosh
  LB  n= 743  sd Δyards 2.175   corr with direct value +0.04 (n=304)   best: Dre Greenlaw, Tyrel Dodson, Tatum Bethune   worst: Anfernee Jennings, Edwin Jackson
  DB  n= 949  sd Δyards 2.202   corr with direct value +0.02 (n=462)   best: Sterling Moore, Kam Chancellor, Alontae Taylor   worst: Omar Bolden, Trae Waynes

['player_id', 'name', 'position', 'group', 'pfr_id', 'rookie_season']
['run', 'complete', 'incomplete', 'sack', 'interception', 'fumble_lost', 'kneel', 'spike', 'penalty', 'penalty_first_down', 'kick_received', 'kick_recovered', 'good', 'no_good', 'timeout_offense', 'timeout_defense']
97 QBs with >=300 dropbacks 2016-24 and an embedding
corr actual comp vs head: 0.57
corr actual sack vs head: 0.35
               name       n      comp    p_comp      sack    p_sack  p_yds_complete      norm
41   Trevor Siemian  1293.0  0.539830  0.198522  0.060325  0.199596       15.476990  7.128830
53      Cooper Rush   567.0  0.578483  0.186606  0.037037  0.217212       15.585009  8.402298
55  Patrick Mahomes  5156.0  0.633631  0.311864  0.042281  0.128033       16.981087  9.846312
65       Josh Allen  4303.0  0.602138  0.314116  0.047409  0.158868       16.514267  8.903187
                name       n      comp    p_comp
9   Ryan Fitzpatrick  1681.0  0.580012  0.145792
56     DeShone Kizer   558.0  0.485663  0.147155
50         PJ Walker   367.0  0.504087  0.155875
32        Geno Smith  1984.0  0.628528  0.293345
55   Patrick Mahomes  5156.0  0.633631  0.311864
65        Josh Allen  4303.0  0.602138  0.314116
```
### Position swap, ae0
```
120 games; one change to the home team's depth chart vs its previous-game lineup (Δ home win, Δ margin in points, 95% CI):
  slot  change player                              Δ win         Δ margin
  QB    best   Josh Allen               +0.059 ± 0.013   +1.59 ± 0.38
  QB    worst  Davis Mills              -0.021 ± 0.013   -0.76 ± 0.39
  QB    new                             -0.128 ± 0.014   -4.48 ± 0.41
  QB    out                             -0.033 ± 0.015   -1.23 ± 0.43
  RB    best   Aaron Jones              -0.000 ± 0.011   -0.13 ± 0.24
  RB    worst  Ezekiel Elliott          +0.004 ± 0.011   -0.07 ± 0.26
  RB    new                             -0.006 ± 0.010   -0.44 ± 0.25
  RB    out                             +0.006 ± 0.011   -0.04 ± 0.26
  WR    best   Mike Evans               -0.005 ± 0.010   -0.16 ± 0.21
  WR    worst  Jerry Jeudy              -0.010 ± 0.010   -0.48 ± 0.24
  WR    new                             -0.010 ± 0.011   -0.36 ± 0.23
  WR    out                             +0.002 ± 0.011   -0.01 ± 0.26
  TE    best   Travis Kelce             +0.025 ± 0.009   +0.63 ± 0.22
  TE    worst  Drew Sample              -0.009 ± 0.011   -0.35 ± 0.23
  TE    new                             -0.017 ± 0.010   -0.75 ± 0.24
  TE    out                             +0.002 ± 0.010   -0.16 ± 0.21
  OL    best   Andrew Wylie             +0.020 ± 0.009   +0.66 ± 0.21
  OL    worst  Robert Jones             -0.008 ± 0.009   -0.50 ± 0.21
  OL    new                             -0.006 ± 0.009   -0.35 ± 0.21
  OL    out                             +0.001 ± 0.009   -0.09 ± 0.24
  DL    best   Nick Bosa                +0.018 ± 0.008   +0.57 ± 0.18
  DL    worst  Travon Walker            -0.016 ± 0.009   -0.71 ± 0.22
  DL    new                             -0.014 ± 0.008   -0.50 ± 0.20
  DL    out                             +0.005 ± 0.009   +0.11 ± 0.21
  LB    best   Myles Garrett            +0.012 ± 0.009   +0.24 ± 0.21
  LB    worst  Bobby Wagner             -0.020 ± 0.009   -0.64 ± 0.19
  LB    new                             +0.003 ± 0.009   -0.03 ± 0.22
  LB    out                             +0.005 ± 0.010   +0.00 ± 0.20
  DB    best   C.J. Gardner-Johnson     +0.012 ± 0.010   +0.38 ± 0.22
  DB    worst  Mike Jackson             -0.001 ± 0.010   -0.12 ± 0.20
  DB    new                             -0.004 ± 0.009   -0.19 ± 0.20
  DB    out                             +0.006 ± 0.009   -0.02 ± 0.21
  K     best   Brandon Aubrey           +0.003 ± 0.009   -0.05 ± 0.18
  K     worst  Jake Moody               +0.003 ± 0.009   -0.01 ± 0.19
  K     new                             -0.003 ± 0.009   -0.15 ± 0.18
  K     out                             -0.005 ± 0.010   -0.18 ± 0.23
  P     best   Corliss Waitman          +0.003 ± 0.009   -0.08 ± 0.22
  P     worst  Ethan Evans              +0.002 ± 0.008   -0.03 ± 0.18
  P     new                             -0.001 ± 0.008   -0.04 ± 0.18
  P     out                             +0.002 ± 0.010   -0.12 ± 0.21

4514 players with 16+ games: each one's own effect on a league-average play vs a replacement-level player (yards per play; defenders: yards taken away)
  QB  n= 122  sd Δyards 1.671   corr with direct value -0.04 (n=74)   best: Patrick Mahomes, Cooper Rush, Mike Glennon   worst: Tony Romo, Mark Sanchez
  RB  n= 420  sd Δyards 1.435   corr with direct value -0.03 (n=166)   best: Elijah McGuire, Michael Burton, Duke Johnson   worst: Darren McFadden, Jakob Johnson
  WR  n= 549  sd Δyards 1.572   corr with direct value +0.03 (n=272)   best: Van Jefferson, Jordy Nelson, Eric Weems   worst: Kayshon Boutte, River Cracraft
  TE  n= 319  sd Δyards 1.367   corr with direct value +0.14 (n=163)   best: Vernon Davis, James Casey, Kendall Blanton   worst: Cole Kmet, Johnny Mundt
  OL  n= 749  sd Δyards 1.582   corr with direct value -0.01 (n=405)   best: John Urschel, Jacques McClendon, Tyreek Burwell   worst: A.Q. Shipley, Matt Nelson
  DL  n= 663  sd Δyards 1.724   corr with direct value -0.01 (n=367)   best: Brandon Thompson, Jabaal Sheard, Desmond Bryant   worst: R.J. McIntosh, Vance Walker
  LB  n= 743  sd Δyards 1.348   corr with direct value +0.09 (n=304)   best: Justin March, Allen Bradford, Dakota Allen   worst: Darius Fleming, Reggie Walker
  DB  n= 949  sd Δyards 1.653   corr with direct value +0.03 (n=462)   best: Chris Maragos, Noah Igbinoghene, Andru Phillips   worst: Omar Bolden, Duke Ihenacho

['player_id', 'name', 'position', 'group', 'pfr_id', 'rookie_season']
['run', 'complete', 'incomplete', 'sack', 'interception', 'fumble_lost', 'kneel', 'spike', 'penalty', 'penalty_first_down', 'kick_received', 'kick_recovered', 'good', 'no_good', 'timeout_offense', 'timeout_defense']
97 QBs with >=300 dropbacks 2016-24 and an embedding
corr actual comp vs head: 0.57
corr actual sack vs head: 0.34
               name       n      comp    p_comp      sack    p_sack  p_yds_complete      norm
41   Trevor Siemian  1293.0  0.539830  0.196086  0.060325  0.205302       15.312172  7.213810
53      Cooper Rush   567.0  0.578483  0.182110  0.037037  0.222606       15.484787  8.514022
55  Patrick Mahomes  5156.0  0.633631  0.313373  0.042281  0.123486       16.846638  9.950808
65       Josh Allen  4303.0  0.602138  0.316284  0.047409  0.157437       16.440445  9.030127
                name       n      comp    p_comp
9   Ryan Fitzpatrick  1681.0  0.580012  0.142403
56     DeShone Kizer   558.0  0.485663  0.146979
50         PJ Walker   367.0  0.504087  0.154537
32        Geno Smith  1984.0  0.628528  0.298566
55   Patrick Mahomes  5156.0  0.633631  0.313373
65        Josh Allen  4303.0  0.602138  0.316284
```
