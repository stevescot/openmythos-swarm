// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ZkPotContributionRegistry
/// @notice Minimal on-chain registry for anchoring verified training receipts and issuing credits.
/// @dev This is intentionally lightweight and dependency-free to simplify early testing.
contract ZkPotContributionRegistry {
    address public owner;

    mapping(address => bool) public coordinators;
    mapping(bytes32 => bool) public anchoredReceipt;
    mapping(address => uint256) public trainingCredits;

    event CoordinatorUpdated(address indexed coordinator, bool allowed);
    event ReceiptAnchored(
        bytes32 indexed receiptHash,
        address indexed worker,
        string workerId,
        uint256 roundId,
        uint256 credits,
        uint256 timestamp
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    modifier onlyCoordinator() {
        require(coordinators[msg.sender], "not coordinator");
        _;
    }

    constructor(address initialCoordinator) {
        owner = msg.sender;
        coordinators[msg.sender] = true;
        if (initialCoordinator != address(0)) {
            coordinators[initialCoordinator] = true;
            emit CoordinatorUpdated(initialCoordinator, true);
        }
    }

    function setCoordinator(address coordinator, bool allowed) external onlyOwner {
        coordinators[coordinator] = allowed;
        emit CoordinatorUpdated(coordinator, allowed);
    }

    /// @notice Anchor a verified off-chain receipt and credit a worker address.
    /// @param receiptHash sha256/canonical hash from off-chain receipt pipeline.
    /// @param worker EVM address to receive credits.
    /// @param workerId Off-chain worker identifier (for indexing/audit).
    /// @param roundId Federated round number.
    /// @param credits Amount of credits to mint in the in-contract ledger.
    function anchorReceipt(
        bytes32 receiptHash,
        address worker,
        string calldata workerId,
        uint256 roundId,
        uint256 credits
    ) external onlyCoordinator {
        require(!anchoredReceipt[receiptHash], "receipt already anchored");

        anchoredReceipt[receiptHash] = true;
        trainingCredits[worker] += credits;

        emit ReceiptAnchored(
            receiptHash,
            worker,
            workerId,
            roundId,
            credits,
            block.timestamp
        );
    }
}
