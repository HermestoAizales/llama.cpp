<script lang="ts">
	import {
		ChatMessageAgenticContent,
		ChatMessageActions,
		ChatMessageStatistics,
		ModelBadge,
		ModelsSelector
	} from '$lib/components/app';
	import { getMessageEditContext } from '$lib/contexts';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { autoResizeTextarea, copyToClipboard, isIMEComposing } from '$lib/utils';
	import { tick } from 'svelte';
	import { fade } from 'svelte/transition';
	import { Check, X, Pin } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { INPUT_CLASSES } from '$lib/constants';
	import { MessageRole, KeyboardKey, ChatMessageStatsView } from '$lib/enums';
	import Label from '$lib/components/ui/label/label.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { ServerModelStatus } from '$lib/enums';

	import { hasAgenticContent } from '$lib/utils';

	interface Props {
		class?: string;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		isLastAssistantMessage?: boolean;
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		messageContent: string | undefined;
		onCopy: () => void;
		onConfirmDelete: () => void;
		onContinue?: () => void;
		onDelete: () => void;
		onEdit?: () => void;
		onForkConversation?: (options: { name: string; includeAttachments: boolean }) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onRegenerate: (modelOverride?: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		showDeleteDialog: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
		textareaElement?: HTMLTextAreaElement;
	}

	let {
		class: className = '',
		deletionInfo,
		isLastAssistantMessage = false,
		message,
		toolMessages = [],
		messageContent,
		onConfirmDelete,
		onContinue,
		onCopy,
		onDelete,
		onEdit,
		onForkConversation,
		onNavigateToSibling,
		onRegenerate,
		onShowDeleteDialogChange,
		showDeleteDialog,
		siblingInfo = null,
		textareaElement = $bindable()
	}: Props = $props();

	// Get edit context
	const editCtx = getMessageEditContext();

	// Local state for assistant-specific editing
	let shouldBranchAfterEdit = $state(false);

	function handleEditKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			event.preventDefault();
			editCtx.save();
		} else if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			editCtx.cancel();
		}
	}

	const isAgentic = $derived(hasAgenticContent(message, toolMessages));
	const hasReasoning = $derived(!!message.reasoningContent);
	const processingState = useProcessingState();

	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());
	let showRawOutput = $state(false);
	let tokenInspectionActive = $state(false);
	let activeStatsView = $state<ChatMessageStatsView>(ChatMessageStatsView.GENERATION);
	let statsContainerEl: HTMLDivElement | undefined = $state();

	function getScrollParent(el: HTMLElement): HTMLElement | null {
		let parent = el.parentElement;
		while (parent) {
			const style = getComputedStyle(parent);
			if (/(auto|scroll)/.test(style.overflowY)) {
				return parent;
			}
			parent = parent.parentElement;
		}
		return null;
	}

	async function handleStatsViewChange(view: ChatMessageStatsView) {
		const el = statsContainerEl;
		if (!el) {
			activeStatsView = view;

			return;
		}

		const scrollParent = getScrollParent(el);
		if (!scrollParent) {
			activeStatsView = view;

			return;
		}

		const yBefore = el.getBoundingClientRect().top;

		activeStatsView = view;

		await tick();

		const delta = el.getBoundingClientRect().top - yBefore;
		if (delta !== 0) {
			scrollParent.scrollTop += delta;
		}

		// Correct any drift after browser paint
		requestAnimationFrame(() => {
			const drift = el.getBoundingClientRect().top - yBefore;

			if (Math.abs(drift) > 1) {
				scrollParent.scrollTop += drift;
			}
		});
	}

	let highlightAgenticTurns = $derived(
		isAgentic &&
			(currentConfig.alwaysShowAgenticTurns || activeStatsView === ChatMessageStatsView.SUMMARY)
	);

	interface LogprobEntry {
		token: string;
		logprob: number;
		top_logprobs?: Array<{ token: string; logprob: number }>;
	}

	let messageLogprobs = $derived<LogprobEntry[]>(
		message.logprobs ? JSON.parse(message.logprobs) : []
	);
	let hasLogprobs = $derived(Array.isArray(messageLogprobs) && messageLogprobs.length > 0);

	// Token inspection popup state (follows mouse cursor, Google Maps style)
	let activePopup = $state<{ idx: number; x: number; y: number; pinned: boolean } | null>(null);
	let popupHoverActive = $state(false);

	function showPopup(e: MouseEvent, idx: number) {
		if (popupHoverActive && activePopup?.pinned) return;
		activePopup = { idx, x: e.clientX, y: e.clientY, pinned: false };
		popupHoverActive = true;
	}

	function movePopup(e: MouseEvent) {
		if (!popupHoverActive || !activePopup || activePopup.pinned) return;
		activePopup = { ...activePopup, x: e.clientX, y: e.clientY };
	}

	function hidePopup() {
		popupHoverActive = false;
		if (activePopup && !activePopup.pinned) {
			activePopup = null;
		}
	}

	function togglePin() {
		if (activePopup) {
			const newPinned = !activePopup.pinned;
			activePopup = { ...activePopup, pinned: newPinned };
			if (!newPinned && !popupHoverActive) {
				activePopup = null;
			}
		}
	}

	function formatToken(t: string): string {
		if (t === ' ') return '\u2423';
		return t.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
	}

	function computeEntropy(data: LogprobEntry): number {
		let h = 0;
		const topLogprobs = data.top_logprobs ?? [];
		const items =
			topLogprobs.length > 0 ? topLogprobs : [{ token: data.token, logprob: data.logprob }];
		for (const t of items) {
			const p = Math.exp(t.logprob);
			if (p > 0 && p <= 1) {
				h -= p * Math.log2(p);
			}
		}
		return h;
	}

	// --- Token inspection: markdown-aware segmentation ---
	interface TokenSegment {
		type: 'text' | 'bold' | 'italic' | 'code' | 'reasoning';
		tokens: Array<{ token: string; logprob: number; idx: number }>;
	}

	function getTokenClass(logprob: number, idx: number): string {
		const prob = Math.exp(logprob);
		const base = 'cursor-help transition-colors';
		const ring = activePopup?.idx === idx ? 'ring-1 ring-blue-400' : '';
		
		// Exclusion/Selection overrides
		if (isTokenExcluded(idx)) {
			return `inline-block rounded-t px-0.5 ${base} ${ring} token-excluded`;
		}
		if (isAlternativeSelected(idx)) {
			return `inline-block rounded-t px-0.5 ${base} ${ring} token-selected`;
		}
		
		const conf =
			prob >= 0.5
				? 'underline-high'
				: prob >= 0.25
					? 'underline-med'
					: 'underline-low';
		return `inline-block rounded-t px-0.5 ${base} ${ring} ${conf}`;
	}

	function getTokenText(logprob: number, idx: number, token: string): string {
		if (isAlternativeSelected(idx)) {
			return selectedAlternative?.altToken ?? token;
		}
		return token;
	}

	function tokenizeMarkdown(msg: DatabaseMessage, tools: DatabaseMessage[]): TokenSegment[] {
		const segments: TokenSegment[] = [];
		const logprobs = messageLogprobs;
		if (!Array.isArray(logprobs) || logprobs.length === 0) {
			segments.push({ type: 'text', tokens: [] });
			return segments;
		}

		let i = 0;

		// Skip reasoning tokens (tokens before the main content)
		// We look for tokens that appear after reasoning markers like <think> or <think>
		let startIdx = 0;
		let content = '';
		for (let idx = 0; idx < logprobs.length; idx++) {
			content += logprobs[idx].token;
		}

		// Detect reasoning section: <think>...</think> and split
		const thinkMatch = content.match(/<think>[\s\S]*?<\/think>/);
		let reasoningTokens: TokenSegment | null = null;

		if (thinkMatch) {
			const thinkStart = thinkMatch.index ?? 0;
			const thinkEnd = thinkStart + thinkMatch[0].length;

			// Count chars to determine token indices
			let charCount = 0;
			let thinkTokenStart = -1;
			let thinkTokenEnd = -1;
			for (let idx = 0; idx < logprobs.length; idx++) {
				const charBefore = charCount;
				charCount += logprobs[idx].token.length;
				if (charBefore <= thinkStart && thinkTokenStart === -1) {
					thinkTokenStart = idx;
				}
				if (charCount >= thinkEnd && thinkTokenEnd === -1) {
					thinkTokenEnd = idx + 1;
					break;
				}
			}

			if (thinkTokenStart >= 0 && thinkTokenEnd > thinkTokenStart) {
				const tokens = logprobs.slice(thinkTokenStart, thinkTokenEnd).map((t, localIdx) => ({
					token: t.token,
					logprob: t.logprob,
					idx: thinkTokenStart + localIdx
				}));
				reasoningTokens = { type: 'reasoning', tokens };
				startIdx = thinkTokenEnd;
			}
		}

		// Parse remaining content for markdown
		let currentIdx = startIdx;
		while (currentIdx < logprobs.length) {
			const token = logprobs[currentIdx].token;

			// Bold: **text**
			if (token === '**' && currentIdx + 1 < logprobs.length) {
				let j = currentIdx + 1;
				const tokens: TokenSegment['tokens'] = [];
				while (j < logprobs.length && logprobs[j].token !== '**') {
					tokens.push({ token: logprobs[j].token, logprob: logprobs[j].logprob, idx: j });
					j++;
				}
				if (j < logprobs.length && tokens.length > 0) {
					segments.push({ type: 'bold', tokens });
					currentIdx = j + 1;
					continue;
				}
			}

			// Italic: *text*
			if (token === '*' && currentIdx + 1 < logprobs.length && logprobs[currentIdx + 1].token !== '*') {
				let j = currentIdx + 1;
				const tokens: TokenSegment['tokens'] = [];
				while (j < logprobs.length && logprobs[j].token !== '*') {
					tokens.push({ token: logprobs[j].token, logprob: logprobs[j].logprob, idx: j });
					j++;
				}
				if (j < logprobs.length && tokens.length > 0) {
					segments.push({ type: 'italic', tokens });
					currentIdx = j + 1;
					continue;
				}
			}

			// Inline code: `text`
			if (token === '`' && currentIdx + 1 < logprobs.length && logprobs[currentIdx + 1].token !== '`') {
				let j = currentIdx + 1;
				const tokens: TokenSegment['tokens'] = [];
				while (j < logprobs.length && logprobs[j].token !== '`') {
					tokens.push({ token: logprobs[j].token, logprob: logprobs[j].logprob, idx: j });
					j++;
				}
				if (j < logprobs.length && tokens.length > 0) {
					segments.push({ type: 'code', tokens });
					currentIdx = j + 1;
					continue;
				}
			}

			// Regular text token
			segments.push({ type: 'text', tokens: [{ token, logprob: logprobs[currentIdx].logprob, idx: currentIdx }] });
			currentIdx++;
		}

		// Prepend reasoning if found
		if (reasoningTokens) {
			segments.unshift(reasoningTokens);
		}

		return segments;
	}

	// Token hover handlers for popup
	function handleTokenHover(e: MouseEvent) {
		const idx = parseInt((e.currentTarget as HTMLElement).getAttribute('data-token-index') ?? '-1');
		if (idx >= 0) showPopup(e, idx);
	}
	function handleTokenLeave() { hidePopup(); }
	function handleTokenClick(e: MouseEvent) {
		const idx = parseInt((e.currentTarget as HTMLElement).getAttribute('data-token-index') ?? '-1');
		if (idx >= 0) showPopup(e, idx);
	}

	// --- Token Exclusion & Selection ---
	let excludedTokens = $state<Set<number>>(new Set());
	let selectedAlternative = $state<{ tokenIdx: number; altToken: string } | null>(null);

	function toggleTokenExclusion(idx: number) {
		if (excludedTokens.has(idx)) {
			excludedTokens.delete(idx);
		} else {
			excludedTokens.add(idx);
		}
		excludedTokens = new Set(excludedTokens); // trigger reactivity
	}

	function selectAlternative(tokenIdx: number, altToken: string) {
		selectedAlternative = { tokenIdx, altToken };
	}

	function isTokenExcluded(idx: number): boolean {
		return excludedTokens.has(idx);
	}

	function isAlternativeSelected(idx: number): boolean {
		return selectedAlternative?.tokenIdx === idx;
	}

	function resetTokenSelections() {
		excludedTokens = new Set();
		selectedAlternative = null;
	}

	let displayedModel = $derived(message.model ?? null);

	let isCurrentlyLoading = $derived(isLoading());
	let isStreaming = $derived(isChatStreaming());
	let hasNoContent = $derived(!message?.content?.trim());
	let isActivelyProcessing = $derived(isCurrentlyLoading || isStreaming);

	let showProcessingInfoTop = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			hasNoContent &&
			!isAgentic &&
			isLastAssistantMessage
	);

	let showProcessingInfoBottom = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			(!hasNoContent || isAgentic) &&
			isLastAssistantMessage
	);

	function handleCopyModel() {
		void copyToClipboard(displayedModel ?? '');
	}

	$effect(() => {
		if (editCtx.isEditing && textareaElement) {
			autoResizeTextarea(textareaElement);
		}
	});

	$effect(() => {
		if (showProcessingInfoTop || showProcessingInfoBottom) {
			processingState.startMonitoring();
		}
	});
</script>

<div
	class="text-md group w-full leading-7.5 {className}"
	role="group"
	aria-label="Assistant message with actions"
>
	{#if showProcessingInfoTop}
		<div class="mt-6 w-full max-w-[48rem]" in:fade>
			<div class="processing-container">
				<span class="processing-text">
					{processingState.getPromptProgressText() ??
						processingState.getProcessingMessage() ??
						'Processing...'}
				</span>
			</div>
		</div>
	{/if}

	{#if editCtx.isEditing}
		<div class="w-full">
			<textarea
				bind:this={textareaElement}
				value={editCtx.editedContent}
				class="min-h-[50vh] w-full resize-y rounded-2xl px-3 py-2 text-sm {INPUT_CLASSES}"
				onkeydown={handleEditKeydown}
				oninput={(e) => {
					autoResizeTextarea(e.currentTarget);
					editCtx.setContent(e.currentTarget.value);
				}}
				placeholder="Edit assistant message..."
			></textarea>

			<div class="mt-2 flex items-center justify-between">
				<div class="flex items-center space-x-2">
					<Checkbox
						id="branch-after-edit"
						bind:checked={shouldBranchAfterEdit}
						onCheckedChange={(checked) => (shouldBranchAfterEdit = checked === true)}
					/>
					<Label for="branch-after-edit" class="cursor-pointer text-sm text-muted-foreground">
						Branch conversation after edit
					</Label>
				</div>
				<div class="flex gap-2">
					<Button class="h-8 px-3" onclick={editCtx.cancel} size="sm" variant="outline">
						<X class="mr-1 h-3 w-3" />
						Cancel
					</Button>

					<Button
						class="h-8 px-3"
						onclick={editCtx.save}
						disabled={!editCtx.editedContent?.trim()}
						size="sm"
					>
						<Check class="mr-1 h-3 w-3" />
						Save
					</Button>
				</div>
			</div>
		</div>
	{:else if message.role === MessageRole.ASSISTANT}
		{#if showRawOutput}
			<pre class="raw-output">{messageContent || ''}</pre>
			{:else if tokenInspectionActive && hasLogprobs}
			<div class="token-inspection">
				<!-- Markdown-aware token rendering with confidence underlines -->
				{#each tokenizeMarkdown(message, toolMessages) as segment, segIdx}
					{#if segment.type === 'bold'}
						<strong>
							{#each segment.tokens as t}
								<span
									class="{getTokenClass(t.logprob, t.idx)}"
									data-token-index={t.idx}
									onmouseenter={handleTokenHover}
									onmouseleave={handleTokenLeave}
									onclick={handleTokenClick}
									oncontextmenu={(e) => { e.preventDefault(); toggleTokenExclusion(t.idx); }}
								>{getTokenText(t.logprob, t.idx, t.token)}</span>
							{/each}
						</strong>
					{:else if segment.type === 'italic'}
						<em>
							{#each segment.tokens as t}
								<span
									class="{getTokenClass(t.logprob, t.idx)}"
									data-token-index={t.idx}
									onmouseenter={handleTokenHover}
									onmouseleave={handleTokenLeave}
									onclick={handleTokenClick}
									oncontextmenu={(e) => { e.preventDefault(); toggleTokenExclusion(t.idx); }}
								>{getTokenText(t.logprob, t.idx, t.token)}</span>
							{/each}
						</em>
					{:else if segment.type === 'code'}
						<code class="token-code">
							{#each segment.tokens as t}
								<span
									class="{getTokenClass(t.logprob, t.idx)}"
									data-token-index={t.idx}
									onmouseenter={handleTokenHover}
									onmouseleave={handleTokenLeave}
									onclick={handleTokenClick}
									oncontextmenu={(e) => { e.preventDefault(); toggleTokenExclusion(t.idx); }}
								>{getTokenText(t.logprob, t.idx, t.token)}</span>
							{/each}
						</code>
					{:else if segment.type === 'reasoning'}
						<div class="reasoning-section">
							<span class="reasoning-label">Reasoning:</span>
							{#each segment.tokens as t}
								<span
									class="{getTokenClass(t.logprob, t.idx)}"
									data-token-index={t.idx}
									onmouseenter={handleTokenHover}
									onmouseleave={handleTokenLeave}
									onclick={handleTokenClick}
									oncontextmenu={(e) => { e.preventDefault(); toggleTokenExclusion(t.idx); }}
								>{getTokenText(t.logprob, t.idx, t.token)}</span>
							{/each}
						</div>
					{:else if segment.type === 'text'}
						{#each segment.tokens as t}
							<span
								class="{getTokenClass(t.logprob, t.idx)}"
								data-token-index={t.idx}
								onmouseenter={handleTokenHover}
								onmouseleave={handleTokenLeave}
								onclick={handleTokenClick}
								oncontextmenu={(e) => { e.preventDefault(); toggleTokenExclusion(t.idx); }}
							>{getTokenText(t.logprob, t.idx, t.token)}</span>
						{/each}
					{/if}
				{/each}

			<!-- Token detail popup (follows cursor) -->
				{#if activePopup && messageLogprobs[activePopup.idx]}
					{@const t = messageLogprobs[activePopup.idx]}
					{@const mainProb = Math.exp(t.logprob)}
					{@const altProps = t.top_logprobs ?? []}
					{@const altItems = altProps.length > 0 ? altProps.slice(0, 8) : []}
					{@const popupIdx = activePopup.idx}
					<div
						class="token-popup"
						style="left: {activePopup.x + 12}px; top: {activePopup.y - 10}px;"
						onmouseenter={() => { popupHoverActive = true; }}
						onmouseleave={() => { popupHoverActive = false; hidePopup(); }}
					>
						<div class="popup-header">
							<span class="popup-token">{formatToken(t.token)}</span>
							<button class="popup-pin-btn" onclick={togglePin} title="Pin popup">
								<Pin class="h-3.5 w-3.5" />
							</button>
						</div>
						<div class="popup-prob">
							Probability: {(mainProb * 100).toFixed(1)}%
							{#if altItems.length > 0}
								<span class="popup-entropy"> | Entropy: {computeEntropy(t).toFixed(2)}</span>
							{/if}
						</div>
						{#if altItems.length > 0}
							<div class="popup-alternatives">
								<div class="popup-alt-header">Alternatives:</div>
								{#each altItems as alt, i (alt.token + i)}
									{@const altProb = Math.exp(alt.logprob)}
									<div
										class="popup-alt-item {altProb >= 0.5 ? 'alt-high' : altProb >= 0.25 ? 'alt-med' : 'alt-low'} {selectedAlternative?.tokenIdx === popupIdx && selectedAlternative?.altToken === alt.token ? 'alt-selected' : ''}"
										onclick={() => selectAlternative(popupIdx, alt.token)}
										title="Click to substitute this token"
									>
										<span class="alt-token">{formatToken(alt.token)}</span>
										<span class="alt-prob">{(altProb * 100).toFixed(1)}%</span>
									</div>
								{/each}
							</div>
						{/if}
						<div class="popup-actions">
							<button
								class="popup-action-btn {isTokenExcluded(popupIdx) ? 'excluded-active' : ''}"
								onclick={() => toggleTokenExclusion(popupIdx)}
								title="Right-click to exclude / click to toggle exclusion"
							>
								{#if isTokenExcluded(popupIdx)}Exclude (active){:else}Exclude{/if}
							</button>
							{#if excludedTokens.size > 0 || selectedAlternative}
								<button class="popup-action-btn popup-reset" onclick={resetTokenSelections}>
									Reset selection
								</button>
							{/if}
						</div>
					</div>
				{/if}
			</div>
		{:else}
			<ChatMessageAgenticContent
				{message}
				{toolMessages}
				isStreaming={isChatStreaming()}
				highlightTurns={highlightAgenticTurns}
			/>
		{/if}
	{:else}
		<div class="text-sm whitespace-pre-wrap">
			{messageContent}
		</div>
	{/if}

	{#if showProcessingInfoBottom}
		<div class="mt-4 w-full max-w-[48rem]" in:fade>
			<div class="processing-container">
				<span class="processing-text">
					{processingState.getPromptProgressText() ??
						processingState.getProcessingMessage() ??
						'Processing...'}
				</span>
			</div>
		</div>
	{/if}

	<div class="info my-6 grid gap-4 tabular-nums">
		{#if displayedModel}
			<div
				bind:this={statsContainerEl}
				class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground"
			>
				{#if isRouter}
					<ModelsSelector
						currentModel={displayedModel}
						disabled={isLoading()}
						onModelChange={async (modelId, modelName) => {
							const status = modelsStore.getModelStatus(modelId);

							if (status !== ServerModelStatus.LOADED) {
								await modelsStore.loadModel(modelId);
							}

							onRegenerate(modelName);
							return true;
						}}
					/>
				{:else}
					<ModelBadge model={displayedModel || undefined} onclick={handleCopyModel} />
				{/if}

				{#if currentConfig.showMessageStats && message.timings && message.timings.predicted_n && message.timings.predicted_ms}
					{@const agentic = message.timings.agentic}
					<ChatMessageStatistics
						promptTokens={agentic ? agentic.llm.prompt_n : message.timings.prompt_n}
						promptMs={agentic ? agentic.llm.prompt_ms : message.timings.prompt_ms}
						predictedTokens={agentic ? agentic.llm.predicted_n : message.timings.predicted_n}
						predictedMs={agentic ? agentic.llm.predicted_ms : message.timings.predicted_ms}
						agenticTimings={agentic}
						onActiveViewChange={handleStatsViewChange}
					/>
				{:else if isLoading() && currentConfig.showMessageStats}
					{@const liveStats = processingState.getLiveProcessingStats()}
					{@const genStats = processingState.getLiveGenerationStats()}
					{@const promptProgress = processingState.processingState?.promptProgress}
					{@const isStillProcessingPrompt =
						promptProgress && promptProgress.processed < promptProgress.total}

					{#if liveStats || genStats}
						<ChatMessageStatistics
							isLive
							isProcessingPrompt={!!isStillProcessingPrompt}
							promptTokens={liveStats?.tokensProcessed}
							promptMs={liveStats?.timeMs}
							predictedTokens={genStats?.tokensGenerated}
							predictedMs={genStats?.timeMs}
						/>
					{/if}
				{/if}
			</div>
		{/if}
	</div>

	{#if message.timestamp && !editCtx.isEditing}
		<ChatMessageActions
			role={MessageRole.ASSISTANT}
			justify="start"
			actionsPosition="left"
			{siblingInfo}
			{showDeleteDialog}
			{deletionInfo}
			{onCopy}
			{onEdit}
			{onRegenerate}
			onContinue={currentConfig.enableContinueGeneration && !hasReasoning ? onContinue : undefined}
			{onForkConversation}
			{onDelete}
			{onConfirmDelete}
			{onNavigateToSibling}
			{onShowDeleteDialogChange}
			showRawOutputSwitch={currentConfig.showRawOutputSwitch}
			rawOutputEnabled={showRawOutput}
			onRawOutputToggle={(enabled) => {
				showRawOutput = enabled;
				if (enabled && tokenInspectionActive) {
					tokenInspectionActive = false;
				}
			}}
			showTokenInspectionSwitch={hasLogprobs}
			tokenInspectionEnabled={tokenInspectionActive}
			onTokenInspectionToggle={(enabled) => {
				tokenInspectionActive = enabled;
				if (enabled && showRawOutput) {
					showRawOutput = false;
				}
			}}
		/>
	{/if}
</div>

<style>
	.processing-container {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}

	.processing-text {
		background: linear-gradient(
			90deg,
			var(--muted-foreground),
			var(--foreground),
			var(--muted-foreground)
		);
		background-size: 200% 100%;
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		animation: shine 1s linear infinite;
		font-weight: 500;
		font-size: 0.875rem;
	}

	@keyframes shine {
		to {
			background-position: -200% 0;
		}
	}

	/* Token inspection - confidence underlines (Approach 4) */
	.token-inspection {
		font-size: 0.875rem;
		line-height: 1.75;
	}

	.token-span {
		position: relative;
		display: inline;
		border-radius: 2px;
		padding: 0 1px;
		cursor: help;
	}

	.token-span::after {
		content: '';
		position: absolute;
		bottom: -1px;
		left: 0;
		right: 0;
		height: 2px;
		border-radius: 1px;
		opacity: 0.4;
	}

	.underline-high::after {
		background-color: #22c55e;
	}

	.underline-med::after {
		background-color: #eab308;
		opacity: 0.5;
	}

	.underline-low::after {
		background-color: #ef4444;
		opacity: 0.5;
	}

	/* Dark mode adjustments */
	:global(.dark) .underline-high::after {
		background-color: #4ade80;
		opacity: 0.35;
	}

	:global(.dark) .underline-med::after {
		background-color: #facc15;
		opacity: 0.45;
	}

	:global(.dark) .underline-low::after {
		background-color: #f87171;
		opacity: 0.45;
	}

	/* Token hover effect */
	.token-span:hover {
		background-color: hsl(var(--muted) / 0.3);
	}

	/* Reasoning section styling */
	.reasoning-section {
		margin-bottom: 0.75rem;
		padding: 0.5rem;
		border-left: 2px solid hsl(var(--muted-foreground) / 0.3);
	}

	.reasoning-label {
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: hsl(var(--muted-foreground));
		display: block;
		margin-bottom: 0.25rem;
	}

	/* Code block styling in token inspection */
	.token-code {
		font-family: ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
		font-size: 0.82em;
		background-color: hsl(var(--muted) / 0.4);
		border-radius: 4px;
		padding: 1px 4px;
	}

	/* Token Exclusion: red strikethrough */
	.token-excluded {
		text-decoration: line-through;
		text-decoration-color: #ef4444;
		text-decoration-thickness: 2px;
		opacity: 0.5;
	}

	/* Token Selection (alternative): green highlight */
	.token-selected {
		background-color: rgba(34, 197, 94, 0.25) !important;
		border-bottom: 2px solid #22c55e !important;
	}

	:global(.dark) .token-selected {
		background-color: rgba(74, 222, 128, 0.3) !important;
	}

	/* Token detail popup */
	.token-popup {
		position: fixed;
		z-index: 9999;
		background: hsl(var(--card));
		color: hsl(var(--foreground));
		border: 1px solid hsl(var(--border));
		border-radius: 8px;
		padding: 8px 12px;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
		min-width: 180px;
		max-width: 280px;
		font-size: 0.75rem;
		pointer-events: auto;
	}

	.popup-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 4px;
	}

	.popup-token {
		font-weight: 600;
		font-family: ui-monospace, monospace;
		font-size: 0.8rem;
	}

	.popup-pin-btn {
		background: none;
		border: none;
		color: hsl(var(--muted-foreground));
		cursor: pointer;
		padding: 2px;
		border-radius: 3px;
		display: flex;
		align-items: center;
	}

	.popup-pin-btn:hover {
		background: hsl(var(--muted) / 0.5);
		color: hsl(var(--foreground));
	}

	.popup-prob {
		color: hsl(var(--muted-foreground));
		margin-bottom: 6px;
	}

	.popup-entropy {
		opacity: 0.7;
	}

	.popup-alternatives {
		border-top: 1px solid hsl(var(--border));
		padding-top: 6px;
		margin-top: 4px;
	}

	.popup-alt-header {
		font-weight: 600;
		color: hsl(var(--muted-foreground));
		margin-bottom: 4px;
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.popup-alt-item {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 2px 6px;
		border-radius: 4px;
		cursor: pointer;
		transition: background 0.1s;
		margin-bottom: 1px;
	}

	.popup-alt-item:hover {
		background: hsl(var(--accent) / 0.3);
	}

	.popup-alt-item.alt-selected {
		background: hsl(var(--accent) / 0.5);
		border-left: 2px solid #22c55e;
	}

	.alt-token {
		font-family: ui-monospace, monospace;
		font-size: 0.75rem;
	}

	.alt-prob {
		font-size: 0.7rem;
		color: hsl(var(--muted-foreground));
		font-variant-numeric: tabular-nums;
	}

	.alt-high .alt-prob { color: #22c55e; }
	.alt-med .alt-prob { color: #eab308; }
	.alt-low .alt-prob { color: #ef4444; }

	.popup-actions {
		display: flex;
		gap: 6px;
		margin-top: 6px;
		border-top: 1px solid hsl(var(--border));
		padding-top: 6px;
	}

	.popup-action-btn {
		flex: 1;
		padding: 4px 8px;
		border-radius: 4px;
		border: 1px solid hsl(var(--border));
		background: hsl(var(--background));
		color: hsl(var(--foreground));
		font-size: 0.7rem;
		cursor: pointer;
		transition: background 0.15s;
	}

	.popup-action-btn:hover {
		background: hsl(var(--accent) / 0.3);
	}

	.popup-action-btn.excluded-active {
		background: rgba(239, 68, 68, 0.15);
		border-color: rgba(239, 68, 68, 0.3);
		color: #ef4444;
	}

	:global(.dark) .popup-action-btn.excluded-active {
		background: rgba(248, 113, 113, 0.2);
		color: #f87171;
	}

	.popup-action-btn.popup-reset {
		color: hsl(var(--muted-foreground));
	}

	.raw-output {
		width: 100%;
		max-width: 48rem;
		margin-top: 1.5rem;
		padding: 1rem 1.25rem;
		border-radius: 1rem;
		background: hsl(var(--muted) / 0.3);
		color: var(--foreground);
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas,
			'Liberation Mono', Menlo, monospace;
		font-size: 0.875rem;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
