/* This is a file for generating "addition" data
 *
 * The language includes two type of characters
 *     Operators: "+", "="
 *     Numbers: "0", "1" (To be extended)
 * The rules (using DCG) can be expressed like this:
 *     Z --> X, +, Y.
 *     true --> X, =, Y, { X = Y }.
 * where X, Y, Z are sequences of {0, 1}, e.g. [1, 0, 1, 1, 1, 0].
 * REMARK: the "+" operator is always parsed from left to right,
 * i.e. a + b + c + d will always be parsed as ((a + b) + c) + d
 */

:- dynamic my_op/3.
:- discontiguous my_op/3.
:- multifile my_op/3.
:- use_module(library(tabling)).
:- use_module(library(apply)).
:- use_module(library(lists)).
:- table abduce_consistent_eqs/1, consistent_digit_rules/1.
:- thread_local my_op/3.

%%==============================
%% ground facts of my_op/3
%% DEFINED BY USER
%%==============================
% my_op(X, Y, Z) means "my_op(X, Y) -> Z"

%my_op([1], [0], [1]).
%my_op([0], [1], [1]).
%my_op([0], [0], [0]).
%my_op([1], [1], [1, 0]). % carry

%%===================================
%% evaluate a mapped symbol sequence
%%===================================
%% assertion of feature
assert_feat(Feature):-
    assertz_list(Feature).
retract_feat:-
    retractall(my_op(_, _, _)).

%% test if equation is legitimate
is_eq(List_of_Terms):-
    phrase(legit_eq, List_of_Terms).

%% parse a list of terms into my_op(X, Y, Z) format
parse_my_op(List_of_Terms, F):-
    ground(List_of_Terms), % ground instance only
    phrase(parse_eq(eq(X, Y, Z)), List_of_Terms), !,
    F = my_op(X, Y, Z).

%% evaluation
eval_eq(Ex, Feature):-
    setup_call_cleanup(
        assert_feat(Feature),
        (parse_my_op(Ex, X),
         call(X), !
        ),
        retract_feat
    ), !.
eval_eq(Ex, Feature):-
    setup_call_cleanup(
        assert_feat(Feature),
        (phrase(equation(X), Ex),
         call(X), !
        ),
        retract_feat
    ), !.

%% abduction
abduce_eq_ops([], [], false):-
    !.
abduce_eq_ops(Ex, [], true):-
    % when feature is empty, only abduce legitimate examples
    is_eq(Ex).
abduce_eq_ops(Ex, Feature, Label):-
    Feature \= [],
    setup_call_cleanup(assert_feat(Feature),
                       (ground(Feature), not(ground(Ex)),
                        bind_eq_1(Ex, Label)
                       ),
                       retract_feat
                      ).

abduce_consistent_eqs(Exs):-
    % abduce a possible formulation
    maplist(parse_eq, Exs, Eqs),
    split_digit_rules(Eqs, Digit_Rules, Ground_Eqs, Var_Eqs),
    % first abduce consistent digit-rules
    abduce_consistent_drs(Digit_Rules),
    %% DEBUG
    % writeln(Digit_Rules),
    % writeln(Ground_Eqs),
    % writeln(Var_Eqs),
    % writeln('================='),
    %% DEBUG end
    % then abduce consistent equations
    setup_call_cleanup(assert_feat(Digit_Rules),
                       (eval_consistent_g_eqs(Ground_Eqs, Invented_Rules),
                        setup_call_cleanup(assert_feat(Invented_Rules),
                                           abduce_consistent_eqs(Var_Eqs,
                                                                 Ground_Eqs),
                                           retract_list(Invented_Rules)
                                          )
                       ),
                       retract_feat), !.

abduce_consistent_eqs_concurrent(Exs):-
    concurrent_maplist(parse_eq, Exs, Eqs),
    split_digit_rules(Eqs, Digit_Rules, Ground_Eqs, Var_Eqs),
    abduce_consistent_drs(Digit_Rules),
    findall([Rules],
            (setup_call_cleanup(assertz_list(Digit_Rules),
                                eval_consistent_g_eqs(Ground_Eqs,
                                                      Invented_Rules),
                                retract_list(Digit_Rules)
                               ),
             append(Digit_Rules, Invented_Rules, Rules_1),
             list_to_set(Rules_1, Rules_2),
             sort(Rules_2, Rules)
            ),
            Rule_Sets_
           ),
    list_to_set(Rule_Sets_, Rule_Sets),
    maplist(append([abduce_consistent_eqs_thread, Var_Eqs, Ground_Eqs]),
            Rule_Sets,
            Call_List
           ),
    maplist(=.., Goals, Call_List),
    first_solution(Var_Eqs, Goals, [local(inf)]).

abduce_consistent_eqs_thread(Var_Eqs, Ground_Eqs, Rules):-
    setup_call_cleanup(assertz_list(Rules),
                       abduce_consistent_eqs_1(Var_Eqs, Ground_Eqs),
                       retract_list(Rules)
                      ), !.
abduce_consistent_eqs_1([], _).
abduce_consistent_eqs_1([E | Equations], Ground_Eqs):-
    % abduce by finding ground examples
    member(E, Ground_Eqs),
    abduce_consistent_eqs_1(Equations, Ground_Eqs).
abduce_consistent_eqs_1([eq(X, Y, Z) | Equations], Ground_Eqs):-
    % abduce by consulting my_op rules
    my_op_c(X, Y, Z),
    abduce_consistent_eqs_1(Equations, [eq(X, Y, Z) | Ground_Eqs]).


abduce_consistent_drs(Digit_Rules):-
    % abduction
    maplist(op_rule, Digit_Rules),
    % constraints
    consistent_digit_rules(Digit_Rules).
consistent_digit_rules(Digit_Rules):-
    consistent_digit_rules(Digit_Rules, Digit_Rules).
consistent_digit_rules([], _).
consistent_digit_rules([H | T], DRs):-
    consistent_digit_rules1(H, DRs),
    consistent_digit_rules(T, DRs).
consistent_digit_rules1(_, []).
consistent_digit_rules1(my_op(X, Y, Z), [my_op(U, V, W) | DRs]):-
    not((X == U, Y == V, not(Z == W))),
    not((X == U, not(Y == V), Z == W)),
    not((not(X == U), Y == V, Z == W)),
    % symmetry
    not((Y == U, X == V, not(Z == W))),
    not((Y == U, not(X == V), Z == W)),
    not((not(Y == U), X == V, Z == W)),
    consistent_digit_rules1(my_op(X, Y, Z), DRs).

% evaluate if ground equations are consistent
eval_consistent_g_eqs([], X):-
    ground(X), !.
eval_consistent_g_eqs([], X):-
    var(X), X = [], !.
eval_consistent_g_eqs([eq(X, Y, Z) | G_eqs], Rules):-
    % no additional rules existed, call directly
    var(Rules), 
    my_op_c(X, Y, Z),
    eval_consistent_g_eqs(G_eqs, Rules).
eval_consistent_g_eqs([eq(X, Y, Z) | G_eqs], Rules):-
    % try with additional digit rules
    setup_call_cleanup(assertz_list(Rules),
                       my_op_c(X, Y, Z),
                       retract_list(Rules)),
    eval_consistent_g_eqs(G_eqs, Rules).
eval_consistent_g_eqs([eq(X, Y, Z) | G_eqs], Rules):-
    % abduce some additional rules
    findall(my_op(U, V, W), my_op(U, V, W), Existed_Rules),
    abduce_op_rules(Rules, Existed_Rules),
    setup_call_cleanup(assertz_list(Rules),
                       my_op_c(X, Y, Z),
                       retract_list(Rules)),
    eval_consistent_g_eqs(G_eqs, Rules).
/*
eval_consistent_g_eqs([]).
eval_consistent_g_eqs([eq(X, Y, Z) | G_eqs]):-
    (my_op_c(X, Y, Z);
     (not(my_op_c(X, Y, _)),
      not(my_op_c(X, _, Z)),
      not(my_op_c(_, Y, Z)))
    ),
    eval_consistent_g_eqs(G_eqs).
*/
% abduce equations that have variable
abduce_consistent_eqs([], _).
abduce_consistent_eqs([E | Equations], Ground_Eqs):-
    % abduce by finding ground examples
    member(E, Ground_Eqs),
    abduce_consistent_eqs(Equations, Ground_Eqs).
abduce_consistent_eqs([eq(X, Y, Z) | Equations], Ground_Eqs):-
    % abduce by consulting my_op rules
    my_op_c(X, Y, Z),
    abduce_consistent_eqs(Equations, [eq(X, Y, Z) | Ground_Eqs]).

abduce_consistent_eqs([eq(X, Y, Z) | Equations], Ground_Eqs):-
    /*
    % abduce by inventing new equation
    digits(X), digits(Y), digits(Z),
    % whether if my_op_c can get answer
    % (digit rule is able to deduce the result)
    (my_op_c(X, Y, Z);
     (not(my_op_c(X, Y, _)),
      not(my_op_c(X, _, Z)),
      not(my_op_c(_, Y, Z)))
    ),
    */
    % invent new my_op which does not conflict with existed ones
    findall(my_op(U, V, W), my_op(U, V, W), Existed_Rules),
    abduce_op_rules(New_Rules, Existed_Rules),
    setup_call_cleanup(assertz_list(New_Rules),
                       my_op_c(X, Y, Z),
                       retract_list(New_Rules)),
    
    % do not conflict exist examples
    not((member(eq(X, Y, U1), Ground_Eqs), not(U1 == Z))),
    not((member(eq(U2, Y, Z), Ground_Eqs), not(U2 == X))),
    not((member(eq(X, U3, Z), Ground_Eqs), not(U3 == Y))),
    % symmetry
    not((member(eq(Y, X, U4), Ground_Eqs), not(U4 == Z))),
    not((member(eq(U5, X, Z), Ground_Eqs), not(U5 == Y))),
    not((member(eq(Y, U6, Z), Ground_Eqs), not(U6 == X))),
    abduce_consistent_eqs(Equations, [eq(X, Y, Z) | Ground_Eqs]).

% abduce non-conflicting op rules
abduce_op_rules([R | Rs], Existed_Rules):-
    abduce_op_rule(R, Existed_Rules),
    abduce_op_rules(Rs, [R | Existed_Rules]).
abduce_op_rules([], _).

abduce_op_rule(my_op([X], [Y], Z), Rules):-
    digit(X), digit(Y),
    between(1, 2, L), length(Z, L), % sum carry at most one bit
    digits(Z),
    % not existed
    not(member(my_op([X], _, Z), Rules)),
    not(member(my_op(_, [Y], Z), Rules)),
    not(member(my_op([X], [Y], _), Rules)),
    not(member(my_op([Y], _, Z), Rules)),
    not(member(my_op(_, [X], Z), Rules)),
    not(member(my_op([Y], [X], _), Rules)),
    % no conflict
    not((member(my_op([X], [Y], Z1), Rules), Z \= Z1)),
    not((member(my_op([Y] ,[X], Z2), Rules), Z \= Z2)),
    not((member(my_op([X1], [Y], Z), Rules), X \= X1)),
    not((member(my_op([Y], [X2], Z), Rules), X \= X2)),
    not((member(my_op([X], [Y1], Z), Rules), Y \= Y1)),
    not((member(my_op([Y2], [X], Z), Rules), Y \= Y2)).

parse_eq(List_of_Terms, Eq):-
    phrase(parse_eq(Eq), List_of_Terms).

% split Eqs to digit rules (my_op([_], [_], _), digit + digit = digits)
% and equations (eq(_, _, _), digits + digits = digits)
split_digit_rules([], [], [], []).
split_digit_rules([D | Eqs], [D_new | Digit_Rules], G_Eqs, V_Eqs):-
    D = eq([X], [Y], Z), length(Z, LZ), LZ =< 2, LZ >= 1,
    D_new = my_op([X], [Y], Z),
    split_digit_rules(Eqs, Digit_Rules, G_Eqs, V_Eqs).
split_digit_rules([E | Eqs], Digit_Rules, [E | G_Eqs], V_Eqs):-
    % ground equations
    ground(E), E = eq(X, Y, _),
    length(X, LX), length(Y, LY),
    (LX > 1; LY > 1),
    split_digit_rules(Eqs, Digit_Rules, G_Eqs, V_Eqs).
split_digit_rules([E | Eqs], Digit_Rules, G_Eqs, [E | V_Eqs]):-
    % ground equations
    not(ground(E)), E = eq(X, Y, _),
    length(X, LX), length(Y, LY),
    (LX > 1; LY > 1),
    split_digit_rules(Eqs, Digit_Rules, G_Eqs, V_Eqs).


% convert a list of term to a conjunction
list_to_conj([H], H) :- !.
list_to_conj([H | T], ','(H, Conj)) :-
    list_to_conj(T, Conj).

% test with binary addition rules
test_eval_eq(List_of_Terms):-
    eval_eq(List_of_Terms, [
                my_op([1], [0], [1]),
                my_op([0], [1], [1]),
                my_op([0], [0], [0]),
                my_op([1], [1], [1, 0])
            ]
           ).

% abduce possible bindings in equation given rules and label
bind_eq(List_of_Terms, Rules, Label):-
    setup_call_cleanup(assert_op_rules(Rules),
                       (ground(Rules), not(ground(List_of_Terms)),
                        remove_nulls(List_of_Terms, Symbol_Seq),
                        bind_eq_1(Symbol_Seq, Label)),
                       retractall(my_op(_, _, _))
                      ).
bind_eq_1(Symbol_Seq, true):-
    phrase(equation(X), Symbol_Seq),
    call(X).
bind_eq_1(Symbol_Seq, false):-
    illegal_bind(Symbol_Seq).
bind_eq_1(Symbol_Seq, false):-
    phrase(equation(X), Symbol_Seq),
    not(call(X)).

test_bind_eq(List_of_Terms, Label):-
    bind_eq(List_of_Terms, [
                my_op(1, 0, [1]),
                my_op(0, 1, [1]),
                my_op(0, 0, [0]),
                my_op(1, 1, [1, 0])
            ],
            Label
           ).

illegal_bind(Symbol_Seq):-
    member(X, Symbol_Seq),
    var(X),
    symb(X),
    not(phrase(equation(_), Symbol_Seq)).

assert_op_rules(Rules):-
    assertz_list(Rules).

assertz_list([]):-
    !.
assertz_list([H | T]):-
    assertz(H),
    assertz_list(T).
retract_list([]):-
    !.
retract_list([H | T]):-
    retract(H),
    retract_list(T).

remove_nulls([], []).
remove_nulls([null | Xs], Rs):-
    remove_nulls(Xs, Rs).
remove_nulls([X | Xs], [X | Rs]):-
    not(X == null),
    remove_nulls(Xs, Rs).

%%============================================================
%% background knowledge for parsing sequences (Do not modify)
%%============================================================
%% symbols to be mapped
digit(0). 
digit(1).
op(=).
op(+).
null(null).

symb(X):-
    digit(X);
    op(X).

%% define equals/2
equals(X, Y):-
    X == Y.

%% abductive calculation with my_op/3
my_op_c(X, Y, Z):-
    is_list(X), is_list(Y), is_list(Z),
    length(X, LX), length(Y, LY), length(Z, LZ),
    M is max(LX, LY), M1 is M + 1,
    between(M, M1, LZ),
    my_op_c_1(X, Y, Z),
    maplist(digits, [X, Y, Z]).
my_op_c(X, Y, Z):-
    is_list(X), is_list(Y), var(Z),
    length(X, LX), length(Y, LY),
    M is max(LX, LY), M1 is M + 1,
    between(M, M1, LZ),
    length(Z, LZ),
    my_op_c_1(X, Y, Z),
    maplist(digits, [X, Y, Z]).
my_op_c(X, Y, Z):-
    is_list(X), is_list(Z), var(Y),
    length(X, LX), length(Z, LZ), LZ >= LX,
    between(LX, LZ, LY),
    length(Y, LY),
    my_op_c_1(X, Y, Z),
    maplist(digits, [X, Y, Z]).
my_op_c(X, Y, Z):-
    is_list(Y), is_list(Z), var(X),
    length(Y, LY), length(Z, LZ), LZ >= LY,
    between(LY, LZ, LX),
    length(X, LX),
    my_op_c_1(X, Y, Z),
    maplist(digits, [X, Y, Z]).
my_op_c_1(X, Y, Z):-
    reverse(X, X1),
    reverse(Y, Y1),
    my_op_r(X1, Y1, Z1, []), % init carry is []
    reverse(Z1, Z).
my_op_r([], [], Carry, Carry):-
    !.
my_op_r(Xs, Ys, Zs, Carry):-
    not(Carry = []), % exist Carry
    my_op_r(Xs, Carry, Xs1, []), % resolve carry
    my_op_r(Xs1, Ys, Zs, []).
my_op_r([X | Xs], [], [X | Zs], []):-
    my_op_r(Xs, [], Zs, []).
my_op_r([], [Y | Ys], [Y | Zs], []):-
    my_op_r([], Ys, Zs, []).
my_op_r([X | Xs], [Y | Ys], [Z | Zs], []):-
    once((my_op([X], [Y], Z0); my_op([Y], [X], Z0))), % symmetry
    length(Z0, 1), % no need for carry
    [Z] = Z0,
    my_op_r(Xs, Ys, Zs, []).
my_op_r([X | Xs], [Y | Ys], [Z | Zs], []):-
    once((my_op([X], [Y], Z0); my_op([Y], [X], Z0))), % symmetry
    length(Z0, L), L > 1, % Z0 contains carry
    append(Carry, [Z], Z0), !,
    reverse(Carry, Carry_r), % carry needs to be reversed, too
    my_op_r(Xs, Ys, Zs, Carry_r).

% candidate rules
op_rules(R):-
    findall(D, digit(D), Ds), length(Ds, L), % number of digits
    Tot is L*L,
    between(1, Tot, N),
    % digit addition combinations
    findall(my_op([X], [Y], _), (digit(X), digit(Y)), Combs),
    length(R, N), my_subset(R, Combs),
    concurrent_maplist(op_rule, R).
op_rule(R):-
    digit(X), digit(Y),
    between(0, 2, L), length(Z, L), % sum carry at most one bit
    digits(Z),
    R = my_op([X], [Y], Z).

rand_op_rules(R):-
    findall(D, digit(D), Ds), length(Ds, L), % number of digits
    Tot is L*L + 1,
    random(1, Tot, Num_Rules), % random rule length
    % digit addition combinations
    findall(my_op([X], [Y], _), (digit(X), digit(Y)), Combs),
    % random select
    rand_subset(R, Combs, Num_Rules),
    concurrent_maplist(rand_op_rule, R), !.
rand_op_rule(my_op([X], [Y], Z)):-
    digit(X), digit(Y),
    random(1, 3, L), length(Z_, L), % sum carry at most one bit
    concurrent_maplist(rand_digit, Z_),
    digits(Z_) -> Z = Z_; rand_op_rule(my_op([X], [Y], Z)).
    
my_subset([], []).
my_subset([X | L], [X | S]) :-
    my_subset(L, S).
my_subset(L, [_ | S]) :-
    my_subset(L, S).

rand_digit(D):-
    findall(X, digit(X), Xs),
    random_select(D, Xs, _).

rand_subset([], _, 0):-
    !.
rand_subset([], [], _):-
    !.
rand_subset([X | Xs], List, N):-
    random_select(X, List, Rest),
    N1 is N - 1,
    rand_subset(Xs, Rest, N1).

%% parser
% digits
digits([D]) --> [D], { digit(D) }. % empty list [] is not a digit
digits([D | T]) --> [D], !, digits(T), { digit(D) }.
digits(X):-
    phrase(digits(X), X),
    length(X, L),
    (L > 1 -> X \= [0 | _]; true).

/*
% equation
equation(equals(L, R)) --> expr(L), [=], expr(R).
% expression
expr(X) --> digits(X), { digits(X) }.
expr(Z) --> digits(X), [+], expr(Y), { digits(X), my_op_c(X, Y, Z) }. % do calculation

% legitimate equation
legit_eq --> legit_expr, [=], legit_expr.
legit_expr --> digits(X), { digits(X) }.
legit_expr --> digits(X), [+], legit_expr, { digits(X) }.
*/

%% only supports X+Y=Z pattern
equation(equals(L, R)) --> expr(L), [=], digits(R).
expr(Z) --> digits(X), [+], digits(Y), { digits(X), digits(Y), my_op_c(X, Y, Z) }.

legit_eq --> digits(X), [+], digits(Y), [=], digits(Z),
             { digits(Z), digits(X), digits(Y) }.

%% abducing equation
eq_arg([D]) --> [D], { not(D == '+'), not(D == '=') }.
eq_arg([D | T]) --> [D], !, eq_arg(T), { not(D == '+'), not(D == '=') }.
parse_eq(eq(X, Y, Z)) -->
    eq_arg(X), [+], eq_arg(Y), [=], eq_arg(Z),
    % rules for argument length
    { length(X, LX), length(Y, LY), length(Z, LZ),
      LZ =< max(LX, LY) + 1, LZ >= max(LX, LY) }.

%% testing
%?- phrase(expr(X), [1, 1, +, 0, +, 1, +, 1, 1, 0]), !.
%@ X = [1, 0, 1, 0].
%?- eval_eq([1, +, 0, +, 1, =, 1, 0]) -> writeln("Bingo!"); writeln("Wrooooong!"), !.
%@ Bingo!
%@ X = equals([1, 0], [1, 0]).
%?- phrase(equation(X), [1, 0, +, 0, +, 1, =, 1, +, 1, 0, 1]), (call(X) -> writeln("Bingo!"); writeln("Wrooooong!")), !.
%@ Wrooooong!
%@ X = equals([1, 1], [1, 1, 0]).
%?- my_op_c([1, 1], [1, 1], R).
%@ R = [1, 1, 0] ;
%@ false.
%?- my_op_c([1, 1], [0, 1], R).
%@ R = [1, 0, 0] ;
%@ false.
%?- is_eq([1,1,0,+,1,1,+,1,=,1,0,1]).
%@ true.
%?- parse_my_op([1,1,0,+,1,1,=,1,0,0,1], X).
%@ X = my_op([1, 1, 0], [1, 1], [1, 0, 0, 1]).
%?- abduce_consistent_eqs([[1,1,_,1,_,_,0,_], [1,_,_,_,1,0], [1,_,1,1,_,1,_,_]]).
%@ Abduced consistent binding: 
%@ [[1,1,+,1,=,1,0,0],[1,+,0,=,1,0],[1,+,1,1,=,1,0,0]]
%@ true ;
%@ Abduced consistent binding: 
%@ [[1,1,+,1,=,1,0,1],[1,+,0,=,1,0],[1,+,1,1,=,1,0,1]]
%@ true ;
%@ Abduced consistent binding: 
%@ [[1,1,+,1,=,1,0,0],[1,+,1,=,1,0],[1,+,1,1,=,1,0,0]]
%@ true ;
%@ Abduced consistent binding: 
%@ [[1,1,+,1,=,1,0,0],[1,+,1,=,1,0],[1,+,1,1,=,1,0,0]]
%@ true ;
%@ Abduced consistent binding: 
%@ [[1,1,+,1,=,1,0,0],[1,+,1,=,1,0],[1,+,1,1,=,1,0,0]]
%@ true ;
%@ Abduced consistent binding: 
%@ [[1,1,+,1,=,1,0,0],[1,+,1,=,1,0],[1,+,1,1,=,1,0,0]]
%@ true ;
%@ false.;
%?- abduce_consistent_eqs([[1,1,_,1,_,_,1,_], [_,_,_,_,_], [1,_,_,_,1,0], [1,_,1,1,_,1,_,_], [1,+,0,=,1], [1,1,+,1,1,=,1,1,0], [0,+,1,=,_]]).
%@ false.
%?- abduce_consistent_eqs([[1,1,_,1,_,_,0,_], [_,_,_,_,_], [1,_,_,_,1,0], [1,_,1,1,_,1,_,_], [1,+,0,=,1], [1,1,+,1,1,=,1,1,0], [0,+,1,=,_]]).
%@ Abduced consistent binding: 
%@ [[1,1,+,1,=,1,0,0],[0,+,0,=,0],[1,+,1,=,1,0],[1,+,1,1,=,1,0,0],[1,+,0,=,1],[1,1,+,1,1,=,1,1,0],[0,+,1,=,1]]
%@ true .
%?- abduce_consistent_eqs([[1,_,_,0,_,_,_,_], [0,_,0,_,_,_], [1,_,_,_,1,0], [1,_,1,1,_,1,_,_], [1,+,0,=,1], [1,1,+,1,1,=,1,1,0], [0,+,1,=,_]]).
%@ Abduced consistent binding: 
%@ [[1,0,+,0,=,1,0,0],[0,+,0,=,1,0],[1,+,1,=,1,0],[1,+,1,1,=,1,0,0],[1,+,0,=,1],[1,1,+,1,1,=,1,1,0],[0,+,1,=,1]]
%@ true ;
%@ Abduced consistent binding: 
%@ [[1,0,+,0,=,1,0,1],[0,+,0,=,1,1],[1,+,1,=,1,0],[1,+,1,1,=,1,0,0],[1,+,0,=,1],[1,1,+,1,1,=,1,1,0],[0,+,1,=,1]]
%@ true .
%?- A = [[1,1,_,1,_,_,0,_], [_,_,_,_,_], [1,_,_,_,1,0], [1,_,1,1,_,1,_,_], [1,+,0,=,1], [1,1,+,1,1,=,1,1,0], [0,+,1,=,_]], abduce_consistent_eqs(A).
