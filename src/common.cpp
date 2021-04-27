#include "common.h"
#include <string>
#include <vector>
#include <iostream>


#ifndef NINFO
std::string prefix = "";
std::vector<std::string> v;
#endif

void print_prefix(std::string s)
{
#ifndef NINFO
  prefix = s;
#endif
}

void print_enter(std::string s)
{
#ifndef NINFO
  cout << "\033[1;32mentering:" << s << "\033[0m" << endl;
  v.push_back(s);
#endif
}

void print_leave()
{
#ifndef NINFO
  cout << "\033[1;31mleaving:" << v.back() << "\033[0m" << endl;
  v.pop_back();
#endif
}

void _print_prefix()
{
#ifndef NINFO
  cout << "\033[1;33m" << prefix;
  for (std::string &i : v)
    cout << i;
#endif
}

void _print_suffix()
{
#ifndef NINFO
  cout << "\033[0m" << endl;
#endif
}

void print(std::string s)
{
#ifndef NINFO
  _print_prefix();
  cout << s;
  _print_suffix();
#endif
}

void print(int a)
{
#ifndef NINFO
  _print_prefix();
  cout << a;
  _print_suffix();
#endif
}
