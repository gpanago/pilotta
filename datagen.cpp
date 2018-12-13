#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <map>
using namespace std;

//#define USE_MAP

struct card {
  int c; // card 7 to 14
  int s; // suit 1 to 4
  int val;
};

struct decl {
  int val; // value of the declaration
  int sval; // used to distinguish declartion of the same value
  bool trump;
};

struct hash_e {
  int played;
  int prevplayed;
  int toplay;
#ifndef USE_MAP
  int value;
#endif
};

bool operator <(const hash_e& i, const hash_e& j)
{
  if(i.played != j.played)
    return i.played < j.played;
  if(i.toplay != j.toplay)
    return i.toplay < j.toplay;
  return i.prevplayed < j.prevplayed;
}

int hash_cards[32];
int hash_toplay[4];

const int hash_size = 1000000;

bool operator >(const decl& i, const decl& j)
{
  if(i.val != j.val)
    return i.val > j.val;
  if(i.sval != j.sval)
    return i.sval > j.sval;
  return i.trump;
}

bool operator >(const card& i, const card& j)
{
  if(i.s != j.s)
    return i.s < j.s;
  if(i.val != j.val)
    return i.val > j.val;
  return i.c > j.c;
}

bool operator ==(const decl& i, const decl& j)
{
  return i.val == j.val && i.sval == j.sval && i.trump == j.trump;
}

class pilotta {
private:
  // sort cards for each player
  void sort()
  {
    for(int i = 0; i < 4; i++)
      std::sort(cards.begin() + i*8, cards.begin() + (i+1)*8,
                [](auto j, auto k) -> bool {
                  if(j.s == k.s)
                    return j.c > k.c;
                  else
                    return j.s < k.s;
                }
                );
    
  }

  // Put the strongest card first followed by the weakest
  // The idea is that usually the strongest play is to play
  // one of the two.
  void sort_play()
  {
    /*
    vector<card> newcards(32);
    vector<bool> avail(32, true);
    int ind = 0;
    for(int i = 0; i < 4; i++) {
      std::sort(cards.begin() + 8*i, cards.begin() + 8*(i+1), std::greater<card>());
      int s = 1;
      while(ind < 8*(i+1)) {
	if(s == 5)
	  s = 1;
	int j = 8*i;
	for(; j < 8*(i+1); j++)
	  if(cards[j].s == s && avail[j]) {
	    avail[j] = false;
	    newcards[ind++] = cards[j];
	    break;
	  }
	if(j == 8*(i+1)) {
	  s++;
	  continue;
	}
	for(; j < 8*(i+1); j++)
	  if(cards[j].s != s || !avail[j])
	    break;
	j--;
	if(cards[j].s == s && avail[j]) {
	  avail[j] = false;
	  newcards[ind++] = cards[j];
	}
	s++;
      }
    }
    for(int i = 0; i < 32; i++)
      cards[i] = newcards[i];
    */
  }
  
  int rec(vector<bool>& inds, vector<card>& prev,
          int toplay, int previ, int sct, int scf,
	  int alpha, int beta,
#ifndef USE_MAP
	  int myhash,
#endif
	  int myplayed, int myprevplayed)
  {
    alpha = max(alpha, sct);
    beta = min(beta, 162 - scf);
    if(previ == 32) {
      // assert(sct + scf == 152);
      if(toplay % 2 == 0) // To piso
        return sct + 10;
      else
        return sct;
    }

    // Search hash
#ifndef USE_MAP
    hash_e& probe = hash[myhash % hash_size];
    if(probe.played == myplayed && probe.prevplayed == myprevplayed &&
       probe.toplay == toplay && probe.value != -1)
      return probe.value + sct;
#else
    hash_e prob_el = {myplayed, myprevplayed, toplay};
    auto probe = hash.find(prob_el);
    if(probe != hash.end())
      return probe->second + sct;
#endif
    bool ourgame = (toplay % 2 == 0);
    int value = (ourgame ? alpha : beta);
    bool moves[8];
    int toplay_n[8], sct_n[8], scf_n[8];
    
    // Are we playing first?
    if(previ % 4 == 0) {
      int toplay_nn = (toplay + 1) % 4;
      for(int i = 8*toplay; i < 8*(toplay+1); i++) {
	int card_i = i % 8;
	if(!inds[i]) {
	  moves[card_i] = false;
	  continue;
	}
	moves[card_i] = true;
	toplay_n[card_i] = toplay_nn;
	sct_n[card_i] = sct;
	scf_n[card_i] = scf;
      }
    } else {
      int first_card = previ - (previ % 4);
      int lead_suit = prev[first_card].s;
      int highest_trump = -1;
      card highest_trump_card;
      for(int i = first_card; i < previ; i++)
        if(prev[i].s == trump && (prev[i].val > highest_trump ||
                                  (prev[i].val == highest_trump &&
                                   prev[i].c > highest_trump_card.c))) {
          highest_trump = prev[i].val;
          highest_trump_card = prev[i];
        }

      bool have_suit = false, have_trump = false, have_higher = false;
      for(int i = 8*toplay; i < 8*toplay + 8; i++) {
	if(!inds[i])
	  continue;
        if(cards[i].s == lead_suit) {
          have_suit = true;
        }
	if(cards[i].s == trump) {
          have_trump = true;
          if(cards[i].val > highest_trump ||
             (cards[i].val == highest_trump && cards[i].c > highest_trump_card.c))
            have_higher = true;
        }
      }
        
      for(int i = 8*toplay; i < 8*(toplay + 1); i++) {
	int card_i = i % 8;
	if(!inds[i]) {
	  moves[card_i] = false;
	  continue;
	}
	// do we have the suit that was lead?
	if(have_suit) {
	  if(cards[i].s != lead_suit) {
	    moves[card_i] = false;
	    continue;
	  }
	} else if(have_trump) {
	  if(cards[i].s != trump) {
	    moves[card_i] = false;
	    continue;
	  }
	}
      
	// we have to play a stronger trump than already played
	// if we have one
	if(cards[i].s == trump && have_higher &&
	   (cards[i].val < highest_trump || (cards[i].val == highest_trump &&
					     cards[i].c < highest_trump_card.c))) {
	  moves[card_i] = false;
	  continue;
	}

	moves[card_i] = true;
	
	// Are we playing last?
	// If so, check who won the trick,
	// count the points, and check
	// who plays next
	if(previ % 4 == 3) {
	  prev[previ] = cards[i];
	  int next;
	  int tscore = trick(prev.begin() + previ - 3, next);
	  toplay_n[card_i] = (next + toplay + 1) % 4;
	  if(toplay_n[card_i] % 2 == 0) {
	    sct_n[card_i] = sct + tscore;
	    scf_n[card_i] = scf;
	  } else {
	    sct_n[card_i] = sct;
	    scf_n[card_i] = scf + tscore;
	  }
	} else {      
	  toplay_n[card_i] = (toplay + 1) % 4;
	  sct_n[card_i] = sct;
	  scf_n[card_i] = scf;
	}
      }
    }
    bool save = false;
#ifndef USE_MAP
    int rawhash = myhash ^ hash_toplay[toplay];
#endif
    for(int i = 0; i < 8 && alpha < beta; i++) {
      if(!moves[i])
	continue;
      int card_ind = 8*toplay + i;
      prev[previ] = cards[card_ind];
      inds[card_ind] = false;
#ifndef USE_MAP
      int newhash = rawhash ^ hash_cards[card_ind] ^ hash_toplay[toplay_n[i]];
#endif
      int newplayed = myplayed ^ (1 << card_ind);
      int newprevplayed;
      if(previ % 4 == 3)
	newprevplayed = 0;
      else
	newprevplayed = myprevplayed ^ (1 << card_ind);
      int newvalue = rec(inds, prev, toplay_n[i],
			 previ + 1, sct_n[i], scf_n[i], alpha, beta,
#ifndef USE_MAP
			 newhash,
#endif
			 newplayed, newprevplayed);
      if(ourgame) {
	if(newvalue > value) {
	  value = newvalue;
	  if(value < beta && value > alpha)
	    save = true;
	  else
	    save = false;
	  alpha = max(value, alpha);
	}
      } else {
	if(newvalue < value) {
	  value = newvalue;
	  if(value < beta && value > alpha)
	    save = true;
	  else
	    save = false;
	  beta = min(beta, value);
	}
      }
      inds[card_ind] = true;
    }

    if(save) {
#ifndef USE_MAP
      probe.played = myplayed;
      probe.prevplayed = myprevplayed;
      probe.toplay = toplay;
      probe.value = value - sct;
#else
      hash[prob_el] = value - sct;
#endif
    }
    return value;
  }


  int trick(const vector<card>::iterator& start, int& next)
  {
    bool trump_played = false;
    int tot = 0;
    for(int i = 0; i < 4; i++) {
      card c = *(start + i);
      tot += c.val;
      if(c.s == trump)
        trump_played = true;
    }
    int highest = -1;
    int lead_suit;
    if(trump_played)
      lead_suit = trump;
    else
      lead_suit = start->s;
    for(int i = 0; i < 4; i++) {
      card c = *(start + i);
      if(c.s == lead_suit) {
        if(c.val > highest) {
          next = i;
          highest = c.val;
        } else if(c.val == highest && c.c > (start + next)->c) {
          next = i;
        }
      }
    }
    return tot;
  }

  void set_val(card& c)
  {
    if(c.s == trump) {
      if(c.c == 9) {
        c.val = 14;
        return;
      } else if(c.c == 11) {
        c.val = 20;
        return;
      }
    }
    switch(c.c) {
    case 7:
    case 8:
    case 9:
      c.val = 0;
      break;
    case 10:
      c.val = 10;
      break;
    case 11:
      c.val = 2;
      break;
    case 12:
      c.val = 3;
      break;
    case 13:
      c.val = 4;
      break;
    case 14:
      c.val = 11;
      break;
    default:;
    }
  }

  string to_card(int c)
  {
    static string cards[] = {"7", "8", "9", "10", "J", "Q", "K", "A"};
    return cards[c - 7];
  }
  
  vector<card> cards;
  vector<decl> decls[4];
#ifdef USE_MAP
  map<hash_e, int> hash;
#else
  vector<hash_e> hash;
#endif
  int trump;

public:

  pilotta(const vector<card>& deck, int s) : cards(deck),
#ifndef USE_MAP
					     hash(hash_size, hash_e({-1, -1, -1, -1})),
#endif
					     trump(s) {
  }
  
  int play(void)
  {
    vector<bool> inds(32);
    vector<card> prevs(32);
    for(int i = 0; i < 32; i++) {
      inds[i] = true;
      set_val(cards[i]);
    }
    sort_play();

    return rec(inds, prevs, 0, 0, 0, 0, 0, 162,
#ifndef USE_MAP
	       0,
#endif
	       0, 0);
  }

  int find_decl(void)
  {
    sort();
    vector<bool> used(32, false);
    for(int i = 0; i < 4; i++) {
      // check for 4 of a kind
      // only for cards from 9 to 14 (A)
      int occurs[6] = {0};
      for(int ind = 8*i; ind < 8*(i+1); ind++)
	if(cards[ind].c >= 9 && cards[ind].c <= 14)
	  occurs[cards[ind].c - 9]++;
      for(int ind = 0; ind < 6; ind++) {
	if(occurs[ind] == 4) {
	  int val;
	  int sval = 0;
	  switch(ind) {
	  case 0:
	    val = 150;
	    break;
	  case 1:
	    val = 100;
	    sval = 45;
	    break;
	  case 2:
	    val = 200;
	    break;
	  default: // 3, 4 or 5
	    val = 100;
	    sval = ind*10;
	  }
	  decls[i].emplace_back(decl({val, sval, false}));
	  for(int j = 8*i; j < 8*(i+1); j++)
	    if(cards[j].c == (ind + 9))
	      used[j] = true;
	}
      }
      card prev = {-1,-1,0};
      int streak = 0;
      for(int ind = 8*i; ind < 8*(i+1); ind++) {
	if(!used[ind] && cards[ind].s == prev.s && prev.c - cards[ind].c == 1 && streak < 5)
	  streak++;
	else {
	  if(streak >= 3) {
	    int val;
	    if(streak == 3)
	      val = 20;
	    else if(streak == 4)
	      val = 50;
	    else
	      val = 100;
	    decls[i].emplace_back(decl({val, prev.c + streak, prev.s == trump}));
	  }
	  if(!used[ind])
	    streak = 1;
	  else
	    streak = 0;
	}
	prev = cards[ind];
      }
      std::sort(decls[i].begin(), decls[i].end(), (bool (*)(const decl&, const decl&)) &(operator>));
    }
    int j = 0;
    vector<int> same_players;
    decl highest;
    for(; j < 4; j++)
      if(decls[j].size() != 0) {
	highest = decls[j][0];
	same_players.push_back(j);
	break;
      }
    if(j == 4) // no one has anything
      return 0;
    for(int k = j+1; k < 4; k++) {
      if(decls[k].empty())
	continue;
      if(decls[k][0] > highest) {
	highest = decls[k][0];
	same_players.resize(0);
	same_players.push_back(k);
      } else if(decls[k][0] == highest)
	same_players.push_back(k);
    }
    bool zero = false, one = false;
    for(int k : same_players)
      if(k % 2 == 0)
	zero = true;
      else
	one = true;
    if(zero && one)
      return 0;
    int tot = 0;
    if(zero) {
      for(j = 0; j < 2; j++)
	for(auto k : decls[2*j])
	  tot += k.val;
    } else {
      for (j = 0; j < 2; j++)
	for(auto k : decls[2*j + 1])
	  tot += k.val;
    }
    if(zero)
      return tot;
    else
      return -tot;
  }

  void write(fstream& out, int s, int d)
  {
    out << "Trump suit: " << trump << endl;
    for(int i = 0; i < 4; i++) {
      out << "Player " << i << " hand:";
      for(int j = i*8; j < (i+1)*8; j++)
	out << ' ' << to_card(cards[j].c) << ',' << cards[j].s;
      out << endl << "Declarations:";
      if(decls[i].size()) {
	for(auto k : decls[i])
	  out << ' ' << k.val << ',' << k.sval << ',' << (k.trump ? 1 : 0);
      } else {
	out << " None";
      }
      out << endl << endl;
    }
    out << "Score: " << s << ". Declaration: " << d << endl << endl << endl;
  }
};
  
int main(int argc, char **argv)
{
  if(argc != 3) {
    cerr << "Usage is: " << argv[0] << " number output\n";
    return 1;
  }
  int num = stoi(argv[1]);
  fstream out(argv[2], fstream::out);
  if(!out.is_open()) {
    cerr << argv[0] << ": Cannot open file " << argv[2] << endl;
    return 1;
  }
  if(num <= 0 || num > 100000000) {
    cerr << argv[0] << ": " << argv[1] << " is not a valid number of data points\n";
    return 1;
  }

  for(auto& k : hash_cards)
    k = rand();
  for(auto& k : hash_toplay)
    k = rand();
  srand(time(NULL));
  vector<card> deck;
  for(int s = 1; s <= 4; s++)
    for(int c = 7; c <= 14; c++)
      deck.emplace_back(card({c, s, 0}));

  int s = 1;
  for(int i = 0; i < num; i++) {
    out << "Data number: " << i << endl;
    random_shuffle(deck.begin(), deck.end());
    vector<card> opposing_deck;
    
    for(int k = 0; k < 8; k++) {
      opposing_deck.push_back(deck[k + 8]);
      opposing_deck.push_back(deck[k + 24]);
    }
      
    for(int j = 1; j <= 20; j++) {
      pilotta p(deck, s);
      int score = p.play();
      int decl = p.find_decl();
      p.write(out, score, decl);
      random_shuffle(opposing_deck.begin(), opposing_deck.end());
      for(int k = 0; k < 8; k++) {
	deck[k + 8] = opposing_deck[k];
	deck[k + 24] = opposing_deck[k + 8];
      }
    }
  }
}
  
