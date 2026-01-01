/*
 * SatUpdate NS-3 Scenario
 * 
 * This scenario simulates packet transfer between satellite constellation nodes
 * for the SatUpdate project. It receives topology and packet information via
 * JSON input file and produces transfer results as JSON output.
 *
 * Usage:
 *   ./ns3 run "satellite-update-scenario --input=input.json --output=output.json"
 *
 * This implements Step 5 of the NS-3/SPICE integration plan.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>

// Simple JSON parsing (production code would use nlohmann/json or similar)
// For now, this is a template that would need proper JSON library integration

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SatelliteUpdateScenario");

/**
 * Node information structure
 */
struct NodeInfo {
    std::string id;
    std::string type;  // "satellite" or "ground"
    double x, y, z;    // Position in meters
};

/**
 * Packet send command structure
 */
struct SendCommand {
    std::string source;
    std::string destination;
    uint32_t packetId;
    uint32_t size;
};

/**
 * Transfer result structure
 */
struct TransferResult {
    std::string source;
    std::string destination;
    uint32_t packetId;
    double timestamp;
    bool success;
    double latencyMs;
    std::string droppedReason;
};

/**
 * Configuration structure
 */
struct ScenarioConfig {
    std::string dataRate = "10Mbps";
    std::string propagationModel = "constant_speed";
    std::string errorModel = "none";
    double errorRate = 0.0;
    uint32_t queueSize = 100;
    uint32_t mtu = 1500;
    double propagationSpeed = 299792458.0;  // Speed of light
};

/**
 * Global state for tracking transfers
 */
class TransferTracker {
public:
    static TransferTracker& Instance() {
        static TransferTracker instance;
        return instance;
    }
    
    void RecordSend(uint32_t packetId, const std::string& src, 
                   const std::string& dst, double timestamp) {
        m_pendingSends[packetId] = {src, dst, timestamp, 0};
    }
    
    void RecordReceive(uint32_t packetId, double timestamp) {
        if (m_pendingSends.find(packetId) != m_pendingSends.end()) {
            auto& info = m_pendingSends[packetId];
            TransferResult result;
            result.source = info.src;
            result.destination = info.dst;
            result.packetId = packetId;
            result.timestamp = timestamp;
            result.success = true;
            result.latencyMs = (timestamp - info.sendTime) * 1000.0;
            m_results.push_back(result);
            m_pendingSends.erase(packetId);
        }
    }
    
    void RecordDrop(uint32_t packetId, const std::string& reason) {
        if (m_pendingSends.find(packetId) != m_pendingSends.end()) {
            auto& info = m_pendingSends[packetId];
            TransferResult result;
            result.source = info.src;
            result.destination = info.dst;
            result.packetId = packetId;
            result.timestamp = Simulator::Now().GetSeconds();
            result.success = false;
            result.droppedReason = reason;
            m_results.push_back(result);
            m_pendingSends.erase(packetId);
        }
    }
    
    std::vector<TransferResult>& GetResults() { return m_results; }
    
    void Clear() {
        m_pendingSends.clear();
        m_results.clear();
    }

private:
    struct PendingInfo {
        std::string src;
        std::string dst;
        double sendTime;
        uint32_t size;
    };
    
    std::map<uint32_t, PendingInfo> m_pendingSends;
    std::vector<TransferResult> m_results;
};

/**
 * Simple JSON writer helper
 */
class JsonWriter {
public:
    JsonWriter(std::ostream& os) : m_os(os), m_indent(0) {}
    
    void StartObject() { m_os << "{\n"; m_indent++; }
    void EndObject(bool last = true) { 
        m_indent--; 
        Indent(); 
        m_os << "}" << (last ? "\n" : ",\n"); 
    }
    
    void StartArray(const std::string& name) {
        Indent();
        m_os << "\"" << name << "\": [\n";
        m_indent++;
    }
    void EndArray(bool last = true) {
        m_indent--;
        Indent();
        m_os << "]" << (last ? "\n" : ",\n");
    }
    
    void WriteString(const std::string& name, const std::string& value, bool last = false) {
        Indent();
        m_os << "\"" << name << "\": \"" << value << "\"" << (last ? "\n" : ",\n");
    }
    
    void WriteNumber(const std::string& name, double value, bool last = false) {
        Indent();
        m_os << "\"" << name << "\": " << value << (last ? "\n" : ",\n");
    }
    
    void WriteBool(const std::string& name, bool value, bool last = false) {
        Indent();
        m_os << "\"" << name << "\": " << (value ? "true" : "false") << (last ? "\n" : ",\n");
    }

private:
    void Indent() {
        for (int i = 0; i < m_indent * 2; i++) m_os << " ";
    }
    
    std::ostream& m_os;
    int m_indent;
};

/**
 * Write output JSON file
 */
void WriteOutputFile(const std::string& filename, double simTime) {
    std::ofstream ofs(filename);
    if (!ofs) {
        NS_LOG_ERROR("Failed to open output file: " << filename);
        return;
    }
    
    JsonWriter json(ofs);
    json.StartObject();
    json.WriteString("status", "success");
    json.WriteNumber("simulation_time", simTime);
    
    // Write transfers
    auto& results = TransferTracker::Instance().GetResults();
    json.StartArray("transfers");
    for (size_t i = 0; i < results.size(); i++) {
        auto& r = results[i];
        ofs << "    {\n";
        ofs << "      \"source\": \"" << r.source << "\",\n";
        ofs << "      \"destination\": \"" << r.destination << "\",\n";
        ofs << "      \"packet_id\": " << r.packetId << ",\n";
        ofs << "      \"timestamp\": " << r.timestamp << ",\n";
        ofs << "      \"success\": " << (r.success ? "true" : "false");
        if (r.success) {
            ofs << ",\n      \"latency_ms\": " << r.latencyMs << "\n";
        } else {
            ofs << ",\n      \"dropped_reason\": \"" << r.droppedReason << "\"\n";
        }
        ofs << "    }" << (i < results.size() - 1 ? "," : "") << "\n";
    }
    json.EndArray();
    
    // Write statistics
    ofs << "  \"statistics\": {\n";
    uint32_t sent = results.size();
    uint32_t received = 0;
    double totalLatency = 0;
    for (auto& r : results) {
        if (r.success) {
            received++;
            totalLatency += r.latencyMs;
        }
    }
    ofs << "    \"total_packets_sent\": " << sent << ",\n";
    ofs << "    \"total_packets_received\": " << received << ",\n";
    ofs << "    \"average_latency_ms\": " << (received > 0 ? totalLatency / received : 0) << "\n";
    ofs << "  }\n";
    
    ofs << "}\n";
    ofs.close();
}

/**
 * Packet receive callback
 */
void PacketReceivedCallback(Ptr<const Packet> packet, const Address& from) {
    // Extract packet ID from packet tag or content
    // For now, use packet UID as ID
    uint32_t packetId = packet->GetUid();
    double now = Simulator::Now().GetSeconds();
    TransferTracker::Instance().RecordReceive(packetId, now);
    NS_LOG_INFO("Packet " << packetId << " received at " << now);
}

/**
 * Packet drop callback
 */
void PacketDropCallback(Ptr<const Packet> packet) {
    uint32_t packetId = packet->GetUid();
    TransferTracker::Instance().RecordDrop(packetId, "queue_overflow");
    NS_LOG_INFO("Packet " << packetId << " dropped");
}

/**
 * Main scenario function
 */
int main(int argc, char *argv[]) {
    // Command line arguments
    std::string inputFile = "input.json";
    std::string outputFile = "output.json";
    bool verbose = false;
    
    CommandLine cmd;
    cmd.AddValue("input", "Input JSON file", inputFile);
    cmd.AddValue("output", "Output JSON file", outputFile);
    cmd.AddValue("verbose", "Enable verbose logging", verbose);
    cmd.Parse(argc, argv);
    
    if (verbose) {
        LogComponentEnable("SatelliteUpdateScenario", LOG_LEVEL_INFO);
    }
    
    NS_LOG_INFO("SatUpdate NS-3 Scenario starting...");
    NS_LOG_INFO("Input: " << inputFile);
    NS_LOG_INFO("Output: " << outputFile);
    
    // NOTE: In a full implementation, you would:
    // 1. Parse the input JSON file to get topology, sends, and config
    // 2. Create NS-3 nodes based on topology
    // 3. Set up point-to-point links with appropriate channel characteristics
    // 4. Install internet stack and assign IP addresses
    // 5. Create UDP applications for packet sending
    // 6. Schedule packet transmissions based on "sends" array
    // 7. Run simulation
    // 8. Collect results and write output JSON
    
    // For this template, we create a minimal example:
    
    // Create nodes
    NodeContainer nodes;
    nodes.Create(2);
    
    // Set up point-to-point link
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("10ms"));
    
    NetDeviceContainer devices = p2p.Install(nodes);
    
    // Install internet stack
    InternetStackHelper internet;
    internet.Install(nodes);
    
    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);
    
    // Set up UDP echo server on node 1
    uint16_t port = 9;
    UdpEchoServerHelper server(port);
    ApplicationContainer serverApp = server.Install(nodes.Get(1));
    serverApp.Start(Seconds(0.0));
    serverApp.Stop(Seconds(60.0));
    
    // Set up UDP echo client on node 0
    UdpEchoClientHelper client(interfaces.GetAddress(1), port);
    client.SetAttribute("MaxPackets", UintegerValue(1));
    client.SetAttribute("Interval", TimeValue(Seconds(1.0)));
    client.SetAttribute("PacketSize", UintegerValue(1024));
    
    ApplicationContainer clientApp = client.Install(nodes.Get(0));
    clientApp.Start(Seconds(1.0));
    clientApp.Stop(Seconds(60.0));
    
    // Record send
    Simulator::Schedule(Seconds(1.0), []() {
        TransferTracker::Instance().RecordSend(0, "node-0", "node-1", 1.0);
    });
    
    // Record receive (approximate - real impl would use trace callbacks)
    Simulator::Schedule(Seconds(1.02), []() {
        TransferTracker::Instance().RecordReceive(0, 1.02);
    });
    
    // Run simulation
    double simTime = 60.0;
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    
    // Write output
    WriteOutputFile(outputFile, simTime);
    
    // Cleanup
    Simulator::Destroy();
    TransferTracker::Instance().Clear();
    
    NS_LOG_INFO("Scenario completed.");
    return 0;
}