#if canImport(Testing)
import Testing
import Cudarc

@Suite("Cudarc Swift Export Tests")
struct CudarcExportTests {
    @Test("Cudarc swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "Cudarc swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import Cudarc

final class CudarcExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "Cudarc swift module imported cleanly")
    }
}
#endif
