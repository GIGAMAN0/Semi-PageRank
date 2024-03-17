import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    # Get the number of outgoing links from the current page
    num_links = len(corpus[page]) if corpus[page] else len(corpus)
    
    # Calculate the probability of choosing each link
    link_prob = damping_factor / num_links if num_links > 0 else 0

    # Calculate the probability of choosing any page at random
    random_prob = (1 - damping_factor) / len(corpus)

    # Create the transition model dictionary
    transition_probs = {p: random_prob for p in corpus}
    transition_probs.update({link: link_prob for link in corpus[page]})

    return transition_probs


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
     # Initialize a dictionary to store the count of visits to each page
    page_rank = {page: 0 for page in corpus}
    
    # Choose a random starting page
    current_page = random.choice(list(corpus.keys()))

    # Simulate the random surfer for n samples
    for _ in range(n):
        # Increment the count for the current page
        page_rank[current_page] += 1
        
        # Update the current page based on the transition model
        transition_probs = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(transition_probs.keys()), weights=transition_probs.values())[0]

    # Normalize the counts to get estimated PageRank values
    total_samples = sum(page_rank.values())
    return {page: rank / total_samples for page, rank in page_rank.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
     # Initialize PageRank values with initial ranks
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    
    # Set a threshold for convergence
    threshold = 0.001

    while True:
        new_page_rank = {}

        # Iteratively calculate new PageRank values
        for page in corpus:
            new_pr = (1 - damping_factor) / num_pages

            for incoming_page, links in corpus.items():
                if page in links:
                    new_pr += damping_factor * page_rank[incoming_page] / len(links)

            new_page_rank[page] = new_pr

        # Check for convergence
        if all(abs(new_page_rank[page] - page_rank[page]) < threshold for page in corpus):
            break

        # Update PageRank values for the next iteration
        page_rank = new_page_rank

    return page_rank


if __name__ == "__main__":
    main()