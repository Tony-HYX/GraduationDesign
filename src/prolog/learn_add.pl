:- ensure_loaded(['add_simulator_thread.pl']).
:- thread_setconcurrency(_, 16).

%%======================
%% predicates for C-API
%%======================
% parse feature from example
parse_feature(Ex, F):-
    parse_my_op(Ex, F).

% test if the instance (after mapping) is legitimage according to KB
legitimate_ex(Ex):-
    is_eq(Ex).

gen_random_feature(F):-
    rand_op_rules(F).

% generate N features
gen_random_features(N, []):-
    N =< 0, !.
gen_random_features(N, [F | Fs]):-
    rand_op_rules(F),
    N1 is N - 1,
    gen_random_features(N1, Fs).

eval_inst_feature(Ex, Feature):-
    eval_eq(Ex, Feature).

abduce_inst_feature(Ex, Feature, Label):-    
    abduce_eq_ops(Ex, Feature, Label).

abduce_consistent_insts(Exs):-
    abduce_consistent_eqs_concurrent(Exs).
    %abduce_consistent_eqs(Exs).
    %call_with_time_limit(1, abduce_consistent_eqs(Exs)).
    %call_with_depth_limit(abduce_consistent_eqs_concurrent(Exs), 32, Re), not(Re = depth_limit_exceeded).
%call_with_inference_limit(abduce_consistent_eqs(Exs), 1, Re), (Re = !; Re = true), !.

consistent_inst_feature(Exs, Feature):-
    maplist(parse_eq, Exs, Eqs),
    split_digit_rules(Eqs, Digit_Rules_, Ground_Eqs, _),
    list_to_set(Digit_Rules_, Digit_Rules),
    setup_call_cleanup(assertz_list(Digit_Rules),
                       eval_consistent_g_eqs(Ground_Eqs, Invented_Rules),
                       retract_list(Digit_Rules)
                      ), !,
    append(Digit_Rules, Invented_Rules, Feature).
    

%%===========
%% primitives
%%===========
prim(create_mapping/2).
prim(subs/3).
prim(eval_eq/2). % evaluate mapped sequence
prim(op_rules/1). 

%[my_op(1, 0, [1]), my_op(0, 1, [1]), my_op(0, 0, [0]), my_op(1, 1, [1, 0])].

%%==========
%% learning
%%==========
a:-
    pos(Pos), neg(Neg),
    learn(Pos, Neg, Prog),
    pprint(Prog).

%%================================================================
%% background knowledge for learning mapping and "addition" rules
%%================================================================
% possible mappings
symb_map(_, 0).
symb_map(_, 1).
symb_map(_, =).
symb_map(_, +).
symb_map(_, null).

% substitution
mapping(Ex, Subs):-
    ground(Ex),
    mapping(Ex, _, Subs).
mapping(Ex, Map, Subs):-
    create_mapping(Ex, Map),
    subs(Ex, Map, Subs), !.
subs([], _, []).
subs([E | Exs], Map, [S | Subs]):-
    member(E-S, Map),
    subs(Exs, Map, Subs).

% create mapping
create_mapping(Ex, Map):-
    ground(Ex),
    list_to_set(Ex, Set),
    maplist(symb_map, Set, Sym),
    list_to_set(Sym, SS), length(Set, L), length(SS, L), % mutually exlusive
    pairs:keys_values_pairs(Set, Sym, Map).

% random mapping
rand_symb_mapping(Ex, Map):-
    ground(Ex),
    list_to_set(Ex, Set),
    findall(S, symb_map(_, S), Symbs),
    rand_pairs(Set, Symbs, Map).

rand_pairs([], _, []):-
    !.
rand_pairs(_, [], []):-
    !.
rand_pairs([L | Ls], Symbs, [L-S | Ms]):-
    random_select(S, Symbs, Symbs1),
    rand_pairs(Ls, Symbs1, Ms).

loop(N, Goal):-
    between(1, N, _),
    call(Goal),
    false.

%?- time(loop(1000,(abduce_consistent_eqs([[1,_,_,0,_,_,_,_], [0,_,0,_,_,_], [1,_,_,_,1,0], [1,_,1,1,_,1,_,_], [1,+,0,=,1], [1,1,+,1,1,=,1,1,0], [0,+,1,=,_]])))).

% 1000 times, 0.022974 per loop;
% 10 times, 0.0227 per loop.
  
%?- gtrace, F = [[1, 1, +, 1, =, 1, 1, 1], [1, +, 0, =, 1], [1, +, 1, =, 1, 1], [1, +, 1, 1, =, 1, 1, 1], [1, +, 0, =, 1], [1, +, 1, 0, 1, =, 1, 1, 1], [0, +, 1, =, 1]], I = [1,1,+,1,=,1,1,1],eval_inst_feature(I, F).


