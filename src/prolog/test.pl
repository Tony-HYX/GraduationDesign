
:- module(test,[abduce_consistent_insts/1, abduce_consistent_inst/3, eval_chess_label_insts/1, eval_chess/1]).

type(2).
type(1).
type(0).

eval_chess([]).
eval_chess([[X,Y,Type]|Others]):-
  eval_chess(Others),
  type(Type),
  noattack([X,Y,Type],Others).

noattack(_,[]).
noattack([X,Y,Type],[[X1,Y1,Type1]|Others]):-
  ((Type=0;Type1=0)->
  (X=\=X1,Y=\=Y1);
  true),
  ((Type=1;Type1=1)->
  (Y1-Y=\=X1-X,Y1-Y=\=X-X1);
  true),
  ((Type=2;Type1=2)->
  (X=\=X1,Y=\=Y1,Y1-Y=\=X1-X,Y1-Y=\=X-X1);
  true),
  noattack([X,Y,Type],Others).


%% eval_chess_label_insts([[[[4,0,0], [6,1,0]],true], [[[1, 1, 2], [3, 3, 2]], false], [[[1, 0, 2], [2, 2, 0]], true], [[[2, 1, 1], [0, 2, 0]], true] ]).
eval_chess_label_insts([]).
eval_chess_label_insts([Parameter|Parameters]):-
    Fun =.. [eval_chess_label|Parameter],
    call(Fun),
    eval_chess_label_insts(Parameters).

%% eval_chess_label([[4,0,0], [6,1,0], [1,2,0], [3,3,1], [5,3,1], [3,6,1], [7,4,2], [0,5,2]],true).
eval_chess_label(Ex,true):-
    ground(Ex),
    eval_chess(Ex).
eval_chess_label(Ex,false):-
    ground(Ex),
    \+ eval_chess(Ex).


% abduce_consistent_insts([[[[4,0,0], [6,1,0], [1,2,0], [3,3,1], [5,3,1], [3,6,X], [7,4,2], [0,5,2]],[X],true] ]).
abduce_consistent_insts([]).
abduce_consistent_insts([Parameter|Parameters]):-
    Fun =.. [abduce_consistent_inst|Parameter],
    call(Fun),
    abduce_consistent_insts(Parameters).

abduce_consistent_inst(Ex,Vars,Label):-
    assign(Vars),
    eval_chess_label(Ex,Label).



assign([]).
assign([Var|Vars]):-
  type(Var),
  assign(Vars).

member(X,[X|_]).
member(X,[_|L]):-
  member(X,L).


% eval_chess([[1,4,2],[2,2,2],[3,7,2],[4,3,2],[5,6,2],[6,8,2],[7,5,2],[8,1,2]]).
