// port-lint: source cufile/safe.rs
package io.github.kotlinmania.cudarc.cufile.safe

import io.github.kotlinmania.cudarc.cufile.result.CufileError
import io.github.kotlinmania.cudarc.cufile.result.driverClose
import io.github.kotlinmania.cudarc.cufile.result.driverOpen
import kotlin.Result

class Cufile {
    companion object {
        fun new(): Result<Cufile> {
            val res = driverOpen()
            return res.map { Cufile() }
        }
    }

    fun close() {
        driverClose()
    }
}
