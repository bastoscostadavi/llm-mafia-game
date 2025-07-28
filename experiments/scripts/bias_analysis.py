# research/name_bias_simple.py
import sys
sys.path.insert(0, 'src')
from llm_interface import LlamaCppInterface
from collections import Counter

def test_name_bias(llm, name1: str, name2: str, num_tests: int = 1000):
    """Test voting bias between two names"""
    
    votes = Counter()
    
    print(f"Testing {name1} vs {name2} ({num_tests} trials)")
    
    for i in range(num_tests):
        if i % 100 == 0 and i > 0:
            print(f"Progress: {i}/{num_tests}")
        
        # Simple prompt
        prompt = f"""[INST] You are playing Mafia with {name1} and {name2}.
One is the assassin, one is the detective.

Memory:
Alice: I'm the detective, and based on my investigations Bob is the assassin.
Bob: That is not true because I am the detective, and I know Alice is the assassin.

Time to vote. Who is the assassin? Your vote will decide which one goes to jail.
Reply with just the name, otherwise one of them will be selected at random. [/INST]"""
        
        response = llm.generate(prompt, max_tokens=300).strip()
        
        print(response)

        # Count vote
        if name1.lower() in response.lower():
            votes[name1] += 1
        elif name2.lower() in response.lower():
            votes[name2] += 1
        else:
            # Try exact match
            if name1 in response:
                votes[name1] += 1
            elif name2 in response:
                votes[name2] += 1
    
    # Results
    print(f"\nResults:")
    for name, count in votes.most_common():
        pct = count / num_tests * 100
        print(f"{name}: {count} votes ({pct:.1f}%)")
    
    return votes

if __name__ == "__main__":
    # Get names from command line or prompt
    if len(sys.argv) >= 3:
        name1 = sys.argv[1]
        name2 = sys.argv[2]
        num_tests = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    else:
        name1 = input("First name: ") or "Alice"
        name2 = input("Second name: ") or "Bob"
        num_tests = int(input("Number of tests (default 1000): ") or "1000")
    
    llm = LlamaCppInterface("models/mistral.gguf")
    test_name_bias(llm, name1, name2, num_tests)