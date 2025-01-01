def loading_bars():
    
    GREEN = "\033[92m"
    RESET = "\033[0m"

    bar_format = f"{GREEN}{{bar}}{GREEN} {RESET} {{l_bar}} {{remaining}} {{postfix}}"
    bar_format_learner = f"{GREEN}{{bar}}{GREEN} {RESET} {{remaining}} {{postfix}}"

    return bar_format, bar_format_learner