
import random
from pathlib import Path
import matplotlib.pyplot as plt

TARGET = "METHINKS IT IS LIKE A WEASEL"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
MUTATION_RATE = 0.05
OFFSPRING_PER_GENERATION = 100
MAX_GENERATIONS = 100000

OUTPUT_DIR = Path(__file__).resolve().parent
random.seed(42)

# Step 1: Fitness function
def fitness(candidate):
    return sum(
        a == b for a, b in zip(candidate, TARGET)
    )

# Step 2: Mutation function
def mutate(parent):
    return "".join(
        random.choice(ALPHABET)
        if random.random() < MUTATION_RATE
        else ch
        for ch in parent
    )

# Step 3: Create a random starting phrase
def random_phrase():
    return "".join(
        random.choice(ALPHABET) for _ in TARGET
    )

# Step 4: Run the evolutionary algorithm
def main():
    parent = random_phrase()
    best_score = fitness(parent)

    scores = [best_score]
    log = [
        f"Generation 0 | Score {best_score}/"
        f"{len(TARGET)} | {parent}"
    ]

    print(log[-1])

    for generation in range(1, MAX_GENERATIONS + 1):

        # Create 100 mutated offspring
        offspring = [
            mutate(parent)
            for _ in range(OFFSPRING_PER_GENERATION)
        ]

        # Find the best offspring
        candidate = max(offspring, key=fitness)
        candidate_score = fitness(candidate)

        # Keep it only if it improves the score
        if candidate_score > best_score:
            parent = candidate
            best_score = candidate_score

        scores.append(best_score)

        line = (
            f"Generation {generation} | "
            f"Score {best_score}/{len(TARGET)} | "
            f"{parent}"
        )

        print(line)
        log.append(line)

        # Stop when the phrase matches perfectly
        if best_score == len(TARGET):
            break

    # Step 5: Save generation results
    (OUTPUT_DIR / "generations.txt").write_text(
        "\n".join(log) + "\n",
        encoding="utf-8"
    )

    # Step 6: Create fitness graph
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(scores)), scores)

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score")
    plt.title("Weasel Program: Evolution of a Phrase")
    plt.ylim(0, len(TARGET) + 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fitness_plot.png",
        dpi=200
    )
    plt.close()

    print(f"\nFinal phrase: {parent}")
    print(f"Final fitness: {best_score}/{len(TARGET)}")
    print("Saved generations.txt and fitness_plot.png")


if __name__ == "__main__":
    main()
