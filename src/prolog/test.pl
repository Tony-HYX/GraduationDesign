what_grade(Other):-
	Grade = Other - 5,
	write(Grade).


/*

solution([]).
solution([X/Y|Others]):-solution(Others),member(X,[1,2,3,4,5,6,7,8]),member(Y,[1,2,3,4,5,6,7,8]),noattack(X/Y,Others).

noattack(_,[]).
noattack(X/Y,[X1/Y1|Others]):-X=\=X1,Y=\=Y1,Y1-Y=\=X1-X,Y1-Y=\=X-X1,noattack(X/Y,Others).




member(X,[X|_]).
member(X,[_|L]):-member(X,L).

delete[S,A/B] :- length(S,9),member(S,[A/B]).

solution(S),length(S,8),S = [1/4,A/2,3/7,4/3,5/6,6/8,7/5,8/1].
*/
