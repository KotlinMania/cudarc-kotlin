pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0" }

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "cudarc-kotlin"

listOf(
    "../half-kotlin",
    "../float8-kotlin",
    "../thiserror-kotlin",
).forEach { path ->
    if (file(path).exists()) {
        includeBuild(path)
    }
}
