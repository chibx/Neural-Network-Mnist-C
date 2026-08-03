CJSON_DIR = "./pkg/cjson"
# @clang -O3 -march=native -flto -funroll-loops -ffast-math -DNDEBUG -lm *.c $(CJSON_DIR)/cJSON.c $(CJSON_DIR)/cJSON_Utils.c -o build/mnist

.PHONY: build
build:
	@clang -O3 -flto -funroll-loops -fassociative-math -fno-signed-zeros -fno-trapping-math -fno-math-errno -DNDEBUG -lm *.c $(CJSON_DIR)/cJSON.c $(CJSON_DIR)/cJSON_Utils.c -o build/mnist
