#include <iostream>
#include <imgDisplay.h>

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s show <image>\n", argv[0]);
        printf("  %s webcam\n", argv[0]);
        printf("  %s blur\n", argv[0]);
        printf("  %s sobel\n", argv[0]);
        return -1;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "show") == 0) {
        if (argc < 3) { printf("Missing image filename\n"); return -1; }
        imgDisplay(argv[2]);
        return 0;
    }

    

    printf("Unknown command: %s\n", cmd);
    return -1;
    return 0;
    // TIP See CLion help at <a href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>. Also, you can try interactive lessons for CLion by selecting 'Help | Learn IDE Features' from the main menu.
}