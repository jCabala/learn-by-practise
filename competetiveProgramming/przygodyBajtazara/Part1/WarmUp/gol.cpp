//https://szkopul.edu.pl/problemset/problem/RDOz-a972ZUQMa34-HwiXwHh/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

const int SIEVE_N = 1000000;

bool isBad(int x, int p) {
    return p > x || x - p == 1 || x - p == 2 || x - p == 4 || x - p == 9 || x - p == 6; 
}

set<int> solve(int x, vector<int>& primes) {
    int i = primes.size() - 1;
    set<int> ans;
    while (x > 0 && i >= 0) {
        if (!isBad(x, primes[i])) {
            x -= primes[i];
            ans.insert(primes[i]);
        }
        i--;
    }
    return ans;
}

vector<int> prime_sieve(int n) {
    vector<int> primes;
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            if (i > 2) {
                 primes.push_back(i);
            }
           
            for (int j = i * 2; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }
    return primes;
}

int main() {
    int t, x;
    cin >> t;
    vector<int> primes = prime_sieve(SIEVE_N);
    
    while(t--) {
        cin >> x;
        auto ans = solve(x, primes);
        cout << ans.size() << "\n";
        int sum = 0;
        for (auto& el : ans) {
            cout << el << " ";
            sum += el;
        }
        //cout << "\n" << "SUM: " << sum;
        cout << "\n";
    }
    return 0;
}