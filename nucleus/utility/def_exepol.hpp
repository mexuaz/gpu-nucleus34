
#ifndef DEF_EXEPOL_HPP_
#define DEF_EXEPOL_HPP_

#ifdef EXE_SEQ
#define EXE_POL std::execution::seq
#define EXE_POL_STR "seq"
#elif EXE_PAR
#define EXE_POL std::execution::par
#define EXE_POL_STR "par"
#elif EXE_PARUNSEQ
#define EXE_POL std::execution::par_unseq
#define EXE_POL_STR "par_unseq"
#elif EXE_UNSEQ
#define EXE_POL std::execution::unseq
#define EXE_POL_STR "unseq"
#endif

#endif // DEF_EXEPOL_HPP_